import os
import json
import cv2
import numpy as np
import imageio as iio

CVtoGL = np.array([
    [1,0,0,0],
    [0,-1,0,0],
    [0,0,-1,0],
    [0,0,0,1]])


class NumpyEncoder(json.JSONEncoder):
    """
        Encoder class for converting numpy arrays into JSON format.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def convert(
        rgbPath: str,
        maskPath: str,
        consistencyPath: str,
        depthPath: str,
        normalPath: str,

        extrPath: str,
        intrPath: str,

        outPath: str,
        imgWidth: int,
        imgHeight: int,
        downScale: float = 1.0
):
    RGB_OUT = '/'.join([outPath, 'images'])
    MASK_OUT = '/'.join([outPath, 'masks'])
    DEPTH_OUT = '/'.join([outPath, 'depth'])
    NORMAL_OUT = '/'.join([outPath, 'normals'])

    # Compute new height
    nWidth, nHeight = int(imgWidth / downScale), int(imgHeight / downScale)

    # (4,4) intrinsics to work with homogenous coordinates
    intrinsics = np.eye(4)

    # Assume one shared intrinsics camera matrix
    intrRaw = np.load(intrPath, allow_pickle=True)[()]
    for key in intrRaw:
        mat = intrRaw[key]
        intrinsics[:3,:3] = mat[:3,:3]

        # Apply downscaling to intrinsics matrix
        # As no cropping takes place fx,fy,cx,cy are simply multiplied with
        # the corresponding factor
        intrinsics[:2, :3] = (1.0 / downScale) * intrinsics[:2, :3]
        break


    for folder in [RGB_OUT, MASK_OUT, DEPTH_OUT, NORMAL_OUT]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    OUTLIER_THRESHOLD = 4
    frames = []

    extrRaw = np.load(extrPath, allow_pickle=True)[()]
    # Process each frame
    for key in extrRaw:

        # matrix in OpenCV format
        extMatrix = extrRaw[key]
        # Convert to OpenGL
        CVtoGL = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # extMatrix = extMatrix.dot(CVtoGL)


        # Read in consistency
        I_CONSIST = '/'.join([consistencyPath, key + '.png'])
        conImg = cv2.imread(I_CONSIST, cv2.IMREAD_GRAYSCALE)
        # Downscale consistency map
        conImg = cv2.resize(conImg, (nWidth, nHeight))
        # We just need one channel

        # Read in depth
        I_DEPTH_FILE = '/'.join([depthPath, key + '.npy'])
        O_DEPTH_FILE = '/'.join([DEPTH_OUT, key + '.npy'])
        depthImg = np.load(I_DEPTH_FILE)
        # Downscale
        outDepth = cv2.resize(depthImg, (nWidth, nHeight))
        # Set depth to the maximum if it's not within all camera views
        outDepth[conImg < OUTLIER_THRESHOLD] = np.max(outDepth)
        # Normalize depth to 0-1 range
        outDepth = (outDepth - np.min(outDepth)) / (np.max(outDepth) - np.min(outDepth))
        assert np.isclose(np.min(outDepth), 0.0) and np.isclose(np.max(outDepth), 1.0)
        assert outDepth.shape == (nHeight, nWidth)
        # Write as .npy file
        np.save(O_DEPTH_FILE, outDepth)

        # Read in normals
        I_NORMAL_FILE = '/'.join([normalPath, key + '.npy'])
        O_NORMAL_FILE = '/'.join([NORMAL_OUT, key + '.npy'])
        normalImg = np.load(I_NORMAL_FILE)
        # Downscale
        outNormal = cv2.resize(normalImg, (nWidth, nHeight))
        # Set normal to zero vector
        outNormal[conImg < OUTLIER_THRESHOLD, :] = 0
        print(np.min(normalImg))
        print(np.max(normalImg))
        # Make sdfstudio data parser happy by reordering
        outNormal = np.transpose(outNormal, axes=(2, 0, 1))
        assert outNormal.shape == (3, nHeight, nWidth)
        # Write as .npy file
        print(outNormal.shape)
        np.save(O_NORMAL_FILE, outNormal)



        I_IMG_FILE = '/'.join([rgbPath, key  + '.png'])     # Input RGB image file
        O_IMG_FILE = '/'.join([RGB_OUT, key + '.png'])      # Output RGB image file

        I_MASK_FILE = '/'.join([maskPath, key + '.png'])    # Mask input file
        O_MASK_FILE = '/'.join([MASK_OUT, key + '.png'])    # Mask output file

        rgbImg = cv2.imread(I_IMG_FILE)
        rgbImgRes = cv2.resize(rgbImg, dsize=(nWidth, nHeight))     # Resized image

        maskImg = cv2.imread(I_MASK_FILE)
        maskImgRes = cv2.resize(maskImg, (nWidth, nHeight), cv2.IMREAD_GRAYSCALE)   # Resized mask
        maskBinRes = cv2.threshold(maskImgRes, 254, 255, cv2.THRESH_BINARY)[1]


        rgbImgRes[maskBinRes == 0] = 255        # Make background white



        cv2.imwrite(O_IMG_FILE, rgbImgRes)      # Write RGB image
        cv2.imwrite(O_MASK_FILE, maskBinRes)    # Write mask image


        # Set parameters in JSON
        currentFrame = dict()
        currentFrame['rgb_path'] = O_IMG_FILE
        currentFrame['foreground_mask'] = O_MASK_FILE
        currentFrame['mono_depth_path'] = O_DEPTH_FILE
        currentFrame['mono_normal_path'] = O_NORMAL_FILE

        currentFrame['camtoworld'] = extMatrix
        currentFrame['intrinsics'] = intrinsics
        frames.append(currentFrame)

        print('Processing: {}'.format(key))


    # JSON object for meta_data
    meta_data = dict()
    meta_data['camera_model'] = 'OPENCV'
    meta_data['height'] = nHeight
    meta_data['width'] = nWidth
    meta_data['worldtogt'] = [
        [1,0,0,0],  # world to gt transformation (useful for evauation)
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]]


    # Tell data parser meta data is available
    meta_data['has_foreground_mask'] = True

    # Even though we don't have pretrained network, we still have estimated from COLMAP
    meta_data['has_mono_prior'] = True



    # Scene box
    aabb = np.array([[-0.3, -0.3, -0.3], [0.3, 0.3, 0.3]])

    scene_box = dict()
    scene_box['near'] = 0.5
    scene_box['far'] = 2.0
    scene_box['radius'] = 1.5
    scene_box['collider_type'] = 'box'
    scene_box['aabb'] = aabb

    meta_data['scene_box'] = scene_box
    meta_data['frames'] = frames

    with open('/'.join([outPath, 'meta_data.json']), 'w', encoding='utf-8') as mFile:
        json.dump(meta_data, mFile, ensure_ascii=False, indent=4, cls=NumpyEncoder)

if __name__ == '__main__':
    RGB_IN_PATH = '/mnt/hdd/data/head_38/frame_00000/images'
    MASK_IN_PATH = '/mnt/hdd/data/head_38/frame_00000/masks'
    CONSISTENCY_PATH = '/mnt/hdd/data/head_38/frame_00000/consistency'
    DEPTH_PATH = '/mnt/hdd/data/head_38/frame_00000/depth'
    NORMAL_PATH = '/mnt/hdd/data/head_38/frame_00000/normals'

    EXTRINSICS = '/mnt/hdd/data/head_38/extrinsics.npy'
    INTRINSICS = '/mnt/hdd/data/head_38/intrinsics.npy'


    OUT_PATH = '/mnt/hdd/data/head_38_preprocessed'
    convert(
        RGB_IN_PATH,
        MASK_IN_PATH,

        CONSISTENCY_PATH,
        DEPTH_PATH,
        NORMAL_PATH,

        EXTRINSICS,
        INTRINSICS,
        OUT_PATH,
        2200,
        3208,
        4.0
    )