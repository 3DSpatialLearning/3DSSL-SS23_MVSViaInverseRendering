import os
import json
import cv2
import numpy as np

"""
    Main preprocessing script for transforming custom data to sdfstudio format.
"""


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


def setMissingValues(inpMap: np.ndarray,
                     conMap: np.ndarray,
                     threshold: int) -> np.ndarray:
    """
        Depending on the number of channels (depth = 1, normals = 3),
        set missing values to invalid values in the map.

        Parameters
        ----------
            inpMap  -       The input map which values should be replaced.
            conMap  -       Consistency map which counts how many camera views
                            matched with a pixel.
            threshold   -   Threshold used in filtering with consistency map

    """
    assert len(conMap.shape) == 2

    # Normal map
    if len(inpMap.shape) == 3:
        assert inpMap.shape[2] == 3
        mask = (inpMap ** 2).sum(2) < 0.01

        # Filter out almost-zero normals
        inpMap[mask, :] = np.nan

        # Use consistency graph for further filtering
        inpMap[conMap < threshold, :] = np.nan

    else:
        # Depth map
        inpMap[inpMap == 0] = np.nan
        inpMap[conMap < threshold] = np.nan
    return inpMap

def smoothNormalMap(
        normalMap: np.ndarray,
        kernelSize: int,
        conMap: np.ndarray,
        thres: int
        ) -> np.ndarray:
    """
        Apply mean-smoothing to normal map. The single axes are normalized by
        using a running mean (convolution with 1-kernel)

        normalMap is of shape (height, width, 3)
    """
    normalMap = setMissingValues(normalMap, conMap, thres)

    '''
    height, width, channels = normalMap.shape
    tiled = normalMap.reshape([height // kernelSize, kernelSize, width // kernelSize, kernelSize, channels])
    tiled = np.transpose(tiled, [0, 2, 4, 1, 3])
    tiled = tiled.reshape((height // kernelSize, width // kernelSize, channels, kernelSize ** 2))
    output = np.nanmean(tiled, -1)

    # Upscale to original size
    n_output = np.zeros(normalMap.shape)
    n_output[:,:,0] = np.kron(np.squeeze(output[:,:,0]), np.ones((kernelSize, kernelSize)))
    n_output[:,:,1] = np.kron(np.squeeze(output[:,:,1]), np.ones((kernelSize, kernelSize)))
    n_output[:,:,2] = np.kron(np.squeeze(output[:,:,2]), np.ones((kernelSize, kernelSize)))
    '''
    normalMap[conMap < thres, :] = 0.0

    KERNEL_SIZE = 5

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE)) / KERNEL_SIZE**2
    n_output = np.zeros(normalMap.shape)
    n_output[:,:,0] = cv2.filter2D(normalMap[:,:,0], -1, kernel)
    n_output[:,:,1] = cv2.filter2D(normalMap[:,:,1], -1, kernel)
    n_output[:,:,2] = cv2.filter2D(normalMap[:,:,2], -1, kernel)

    n_output[conMap < thres,:] = 0.0
    return n_output

def downscaleMap(inpMap: np.ndarray,
                 conMap: np.ndarray,
                 threshold: int,
                 downScale: int = 1) -> np.ndarray:
    # Normals shape: (H,W,3)
    # Depth shape: (H,W) -- reshape --> (H,W,1)
    inpMap = setMissingValues(inpMap, conMap, threshold)
    isDepth = len(inpMap.shape) == 2

    if isDepth:
        inpMap = np.expand_dims(inpMap, 2)

    # Apply mean on neighborhood for downsampling
    # Essentially performing convolution with kernel size factor x factor
    factor = int(downScale)
    height, width, channels = inpMap.shape
    tiled = inpMap.reshape([height // factor, factor, width // factor, factor, channels])
    tiled = np.transpose(tiled, [0, 2, 4, 1, 3])
    tiled = tiled.reshape((height // factor, width // factor, channels, factor ** 2))
    output = np.nanmean(tiled, -1)
    if isDepth:
        output = output[:,:,0]
    if not isDepth:
        # Normalize normals
        output = output / (output**2).sum(2, keepdims=True)

    # Set invalid normals/depth values to 0.0
    # Important: Later changing normals, e.g. n = 2*(n+1) needs to be considered
    output = np.nan_to_num(output, 0.0)
    return output

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
        downScale: int = 1,
        smoothing: bool = True,
        depth: bool = False,
        normal: bool = False,
        mask: bool = False
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

    # Threshold for filtering with consistency map
    CONSISTENCY_THRESHOLD = 4

    frames = []
    extrRaw = np.load(extrPath, allow_pickle=True)[()]

    # Process each frame
    for key in extrRaw:
        currentFrame = dict()

        # matrix in OpenCV format
        extMatrix = extrRaw[key]

        # Convert to OpenGL
        # Uncomment if you know that your input data is in OpenCV format
        # CVtoGL = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # As the input here is (mostly) in nerfstudio format, it is known that the extrinsics are in OpenGL format;
        # no conversion needed
        # extMatrix = extMatrix.dot(CVtoGL)

        # Read in consistency
        I_CONSIST = '/'.join([consistencyPath, key + '.png'])
        conImg = cv2.imread(I_CONSIST, cv2.IMREAD_GRAYSCALE)

        if not conImg is None and len(conImg.shape) == 3:
            conImg = conImg[:,:,0]

        # We need to have one channel
        if not conImg is None:
            assert len(conImg.shape) == 2
        else:
            conImg = np.ones(rgbImg.shape[:2])

        # No more than the provided number of cameras should be counted in consistency map
        assert np.max(conImg) <= len(extrRaw)

        # Read in depth
        if depth:
            I_DEPTH_FILE = '/'.join([depthPath, key + '.npy'])
            O_DEPTH_FILE = '/'.join([DEPTH_OUT, key + '.npy'])

            if os.path.isfile(I_DEPTH_FILE):
                depthImg = np.load(I_DEPTH_FILE)
            else:
                depthImg = np.zeros(rgbImg.shape[:2])

            depthFiltered = downscaleMap(depthImg, conImg, CONSISTENCY_THRESHOLD, downScale)
            assert depthFiltered.shape == (nHeight, nWidth)

            np.save(O_DEPTH_FILE, depthFiltered)
            currentFrame['mono_depth_path'] = O_DEPTH_FILE
            currentFrame['sensor_depth_path'] = O_DEPTH_FILE

            # Save depth map additionally as PNG for manually inspecting result
            # Put depth into range [0, 1]
            depthFiltered = depthFiltered / np.max(depthFiltered)
            depthFilteredOut = (255.0 * depthFiltered).astype(np.uint8)
            cv2.imwrite(O_DEPTH_FILE[:-4] + '.png', depthFilteredOut)

        if normal:
            # Read in normals
            I_NORMAL_FILE = '/'.join([normalPath, key + '.npy'])
            O_NORMAL_FILE = '/'.join([NORMAL_OUT, key + '.npy'])

            if os.path.isfile(I_NORMAL_FILE):
                normalImg = np.load(I_NORMAL_FILE)
            else:
                normalImg = np.zeros(rgbImg.shape)

            normalFiltered = downscaleMap(normalImg, conImg, CONSISTENCY_THRESHOLD, downScale)

            # normals need to have shape (3,h,w); (h,w,3) -> (3,h,w)
            normalFiltered = np.transpose(normalFiltered, [2,0,1])
            assert normalFiltered.shape == (3, nHeight, nWidth)
            np.save(O_NORMAL_FILE, normalFiltered)
            currentFrame['mono_normal_path'] = O_NORMAL_FILE

            # Normals are normalized vectors in the range [-1.0, 1.0]^3
            # Convert to values in range [0.0, 1.0]^3
            # Convert back (3,h,w) -> (h,w,3)
            oNormals = 0.5 * (normalFiltered.transpose([1,2,0]) + 1.0)
            imgNormals = (255.0 * oNormals).astype(np.uint8)
            cv2.imwrite(O_NORMAL_FILE[:-4] + '.png', imgNormals)


        I_IMG_FILE = '/'.join([rgbPath, key  + '.png'])     # Input RGB image file
        O_IMG_FILE = '/'.join([RGB_OUT, key + '.png'])      # Output RGB image file

        I_MASK_FILE = '/'.join([maskPath, key + '.png'])    # Mask input file
        O_MASK_FILE = '/'.join([MASK_OUT, key + '.png'])    # Mask output file

        rgbImg = cv2.imread(I_IMG_FILE)
        rgbImgRes = cv2.resize(rgbImg, dsize=(nWidth, nHeight))     # Resized image

        if mask:
            maskImg = cv2.imread(I_MASK_FILE)
            maskImgRes = cv2.resize(maskImg, (nWidth, nHeight), cv2.IMREAD_GRAYSCALE)   # Resized mask
            maskBinRes = cv2.threshold(maskImgRes, 254, 255, cv2.THRESH_BINARY)[1]


            rgbImgRes[maskBinRes == 0] = 255        # Make background white

        cv2.imwrite(O_IMG_FILE, rgbImgRes)      # Write RGB image

        if mask:
            cv2.imwrite(O_MASK_FILE, maskBinRes)    # Write mask image


        # Set parameters in JSON
        currentFrame['rgb_path'] = O_IMG_FILE

        if mask:
            currentFrame['foreground_mask'] = O_MASK_FILE

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
        # world to gt transformation
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]]

    # Tell data parser meta data is available
    meta_data['has_foreground_mask'] = mask

    meta_data['has_mono_prior'] = depth
    meta_data['has_sensor_depth'] = depth

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




def convert_new(
    intrPath: str,
    extrPath: str,
    rgbPath: str,
    depthPath: str,
    normalPath: str,
    maskPath: str,
    outPath: str,
    imgWidth: int,
    imgHeight: int,
    downScale: int,
    depth: bool = False,
    normal: bool = False,
    mask: bool = False
):
    """
        New data conversion function.
        Updated format from Simon. No consistency maps are present.
    """

    RGB_OUT = '/'.join([outPath, 'images'])
    MASK_OUT = '/'.join([outPath, 'masks'])
    DEPTH_OUT = '/'.join([outPath, 'depth'])
    NORMAL_OUT = '/'.join([outPath, 'normals'])

    # Compute new height
    nWidth, nHeight = int(imgWidth / downScale), int(imgHeight / downScale)

    # (4,4) intrinsics to work with homogenous coordinates
    intrinsics = np.eye(4)

    # Assume one shared intrinsics camera matrix
    intrRaw = np.load(intrPath)
    for key in intrRaw:
        mat = intrRaw[key]
        intrinsics[:3, :3] = mat[:3, :3]

        # Apply downscaling to intrinsics matrix
        # As no cropping takes place fx,fy,cx,cy are simply multiplied with
        # the corresponding factor
        intrinsics[:2, :3] = (1.0 / downScale) * intrinsics[:2, :3]
        break

    for folder in [RGB_OUT, MASK_OUT, DEPTH_OUT, NORMAL_OUT]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    frames = []
    extrRaw = np.load(extrPath)

    # Process each frame
    for key in extrRaw:
        currentFrame = dict()

        # matrix in OpenCV format
        extMatrix = extrRaw[key]

        # Read in depth
        if depth:
            I_DEPTH_FILE = '/'.join([depthPath, 'cam_' + key + '.npz'])
            O_DEPTH_FILE = '/'.join([DEPTH_OUT, 'cam_' + key + '.npy'])

            if os.path.isfile(I_DEPTH_FILE):
                depthImg = np.load(I_DEPTH_FILE)['arr_0']
            else:
                depthImg = np.zeros(rgbImg.shape[:2])

            CONSISTENCY_THRESHOLD = 2
            conImg = CONSISTENCY_THRESHOLD * np.ones((int(imgHeight), int(imgWidth)))
            depthFiltered = downscaleMap(depthImg, conImg, CONSISTENCY_THRESHOLD, downScale)
            assert depthFiltered.shape == (nHeight, nWidth)

            np.save(O_DEPTH_FILE, depthFiltered)
            currentFrame['mono_depth_path'] = O_DEPTH_FILE
            currentFrame['sensor_depth_path'] = O_DEPTH_FILE

            # Save depth map additionally as PNG for manually inspecting result
            # Put depth into range [0, 1]
            depthFiltered = depthFiltered / np.max(depthFiltered)
            depthFilteredOut = (255.0 * depthFiltered).astype(np.uint8)
            cv2.imwrite(O_DEPTH_FILE[:-4] + '.png', depthFilteredOut)

        if normal:
            # Read in normals
            I_NORMAL_FILE = '/'.join([normalPath, 'cam_' + key + '.npz'])
            O_NORMAL_FILE = '/'.join([NORMAL_OUT, 'cam_' + key + '.npy'])

            if os.path.isfile(I_NORMAL_FILE):
                normalImg = np.load(I_NORMAL_FILE)['arr_0']
            else:
                normalImg = np.zeros(rgbImg.shape)

            normalFiltered = downscaleMap(normalImg, conImg, CONSISTENCY_THRESHOLD, downScale)

            # normals need to have shape (3,h,w); (h,w,3) -> (3,h,w)
            normalFiltered = np.transpose(normalFiltered, [2, 0, 1])
            assert normalFiltered.shape == (3, nHeight, nWidth)
            np.save(O_NORMAL_FILE, normalFiltered)
            currentFrame['mono_normal_path'] = O_NORMAL_FILE

            # Normals are normalized vectors in the range [-1.0, 1.0]^3
            # Convert to values in range [0.0, 1.0]^3
            # Convert back (3,h,w) -> (h,w,3)
            oNormals = 0.5 * (normalFiltered.transpose([1, 2, 0]) + 1.0)
            imgNormals = (255.0 * oNormals).astype(np.uint8)
            cv2.imwrite(O_NORMAL_FILE[:-4] + '.png', imgNormals)

        I_IMG_FILE = '/'.join([rgbPath, 'cam_' + key + '.png'])  # Input RGB image file
        O_IMG_FILE = '/'.join([RGB_OUT, 'cam_' + key + '.png'])  # Output RGB image file

        I_MASK_FILE = '/'.join([maskPath, 'cam_' + key + '.png'])  # Mask input file
        O_MASK_FILE = '/'.join([MASK_OUT, 'cam_' + key + '.png'])  # Mask output file

        rgbImg = cv2.imread(I_IMG_FILE)
        rgbImgRes = cv2.resize(rgbImg, dsize=(nWidth, nHeight))  # Resized image

        if mask:
            maskImg = cv2.imread(I_MASK_FILE)
            maskImgRes = cv2.resize(maskImg, (nWidth, nHeight), cv2.IMREAD_GRAYSCALE)  # Resized mask
            maskBinRes = cv2.threshold(maskImgRes, 254, 255, cv2.THRESH_BINARY)[1]

            rgbImgRes[maskBinRes == 0] = 255  # Make background white

        cv2.imwrite(O_IMG_FILE, rgbImgRes)  # Write RGB image

        if mask:
            cv2.imwrite(O_MASK_FILE, maskBinRes)  # Write mask image

        # Set parameters in JSON
        currentFrame['rgb_path'] = O_IMG_FILE

        if mask:
            currentFrame['foreground_mask'] = O_MASK_FILE

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
        # world to gt transformation
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]

    # Tell data parser meta data is available
    meta_data['has_foreground_mask'] = mask

    meta_data['has_mono_prior'] = depth
    meta_data['has_sensor_depth'] = depth

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

def new_format():
    """
        New format of data (Simon). No consistency map is available.
    """
    RGB_IN_PATH = '/mnt/hdd/data/data_sebastian/id150/frame_00000/images-2x'

    DEPTH_PATH = '/mnt/hdd/data/data_sebastian/id150/frame_00000/colmap/depth_maps_geometric/16'
    NORMAL_PATH = '/mnt/hdd/data/data_sebastian/id150/frame_00000/colmap/normal_maps_geometric/16'
    MASK_PATH = '/mnt/hdd/data/data_sebastian/id150/frame_00000/alpha_map'

    INTRINSICS = '/mnt/hdd/data/data_sebastian/id150/intrinsics.npz'
    EXTRINSICS = '/mnt/hdd/data/data_sebastian/id150/c2ws.npz'

    OUT_PATH = '/mnt/hdd/data/head_150_frame_00000'

    convert_new(
        INTRINSICS,
        EXTRINSICS,
        RGB_IN_PATH,
        DEPTH_PATH,
        NORMAL_PATH,
        MASK_PATH,
        OUT_PATH,
        int(2200/2),
        int(3208/2),
        1,
        depth=True,
        normal=True,
        mask=True
    )

if __name__ == '__main__':

    new_format()

    '''
    RGB_IN_PATH = '/mnt/hdd/data/head_85/frame_00150/images'
    MASK_IN_PATH = '/mnt/hdd/data/head_85/frame_00150/masks'
    CONSISTENCY_PATH = '/mnt/hdd/data/head_85/frame_00150/consistency'
    DEPTH_PATH = '/mnt/hdd/data/head_85/frame_00150/depth'
    NORMAL_PATH = '/mnt/hdd/data/head_85/frame_00150/normals'

    EXTRINSICS = '/mnt/hdd/data/head_85/extrinsics.npy'
    INTRINSICS = '/mnt/hdd/data/head_85/intrinsics.npy'
    OUT_PATH = '/mnt/hdd/data/head_85_frame_150_ds_005'

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
        downScale=2,
        smoothing=True,
        depth=True,
        normal=True,
        mask=True
    )
    '''