import os
import json
import re

import numpy as np
import imageio as iio
from PIL import Image

# Matrix to convert extrinsic matrix from OpenCV convention to OpenGL convention
# Use:
#   extrinsics_in_GL_format = extrinsics_in_CV_format.dot(CVtoGL)
CVtoGL = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

# Matrix used to convert extrinsics matrix from OpenGL convention to OpenCV convention
# Use:
#   extrinsics_in_CV_format = extrinsics_in_GL_format.dot(GLtoCV)
GLtoCV = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])


class NumpyEncoder(json.JSONEncoder):
    """
        Encoder class for converting numpy arrays into JSON format
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_transform_matrix_idx(jsonObj, index):
    for idx, element in enumerate(jsonObj['frames']):
        path = element['file_path']
        if int(re.findall('\d+', path.split('/')[-1])[0]) == index:
            return idx

"""
    Process raw RGB input data and possibly additional data (such as masks)
    into separate location. The images may be downscaled and information about
    the scene is stored in a JSON file which can be processed by nerfstudio.
"""
def nerfstudio_to_masked(
        rgb_path: str,
        transform_path: str,
        out_path: str,
        mask_path: str = '',
        use_masks: bool = False,
        down_scale: int = 1
):
    """
        Transform a given dataset with extrinsics/intrinsics provided in transform.json into another representation,
        possibly with mask supervision.

        Parameters
        ----------
            rgb_path        - Path to RGB images.
            transform_path  - Path to transforms.json file that contains camera extrinsic/intrinsic parameters.
                              The filenames need to be the same as in rgb/mask path.
            out_path        - Path were the preprocessed images are stored
            mask_path       - Path to masks if used
            use_masks       - Whether to use binary mask supervision or not
            down_scale      - Factor by how much images are downscaled. Value of '1' leaves original image size
    """

    # Create output folder if not exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    OUT_IMG_PATH = '/'.join([out_path, 'images'])
    OUT_MASK_PATH = '/'.join([out_path, 'masks'])

    for dir in [OUT_IMG_PATH, OUT_MASK_PATH]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Iterate over all input images
    arr = os.listdir(rgb_path)

    # Sorting is very important for the transform.json in order to map frames correctly
    sorted(arr)

    tJSON = None

    with open('/'.join([transform_path, 'transforms.json']), 'r') as f:
        tJSON = json.load(f)

    idx = 0

    # Read intrinsic parameters
    outJson = dict()
    outJson['fl_x'] = tJSON['fl_x'] / down_scale
    outJson['fl_y'] = tJSON['fl_y'] / down_scale
    outJson['cx'] = tJSON['cx'] / down_scale
    outJson['cy'] = tJSON['cy'] / down_scale
    outJson['w'] = int(tJSON['w'] / down_scale)
    outJson['h'] = int(tJSON['h'] / down_scale)
    outJson['k1'] = tJSON['k1']
    outJson['k2'] = tJSON['k2']
    outJson['p1'] = tJSON['p1']
    outJson['p2'] = tJSON['p2']
    outJson['frames'] = []

    print('Output size: Width: {}, Height: {}'.format(outJson['w'], outJson['h']))

    print(len(tJSON['frames']))

    for file in arr:
        # If file is image
        if file.endswith('.png') or file.endswith('.jpg'):

            print('INDEX: {}'.format(idx))

            IN_FILE = '/'.join([rgb_path, file])
            RGB_OUT_FILE = '/'.join([OUT_IMG_PATH, file])
            img = iio.v3.imread(IN_FILE)

            nWidth, nHeight = int(img.shape[1] / down_scale), int(img.shape[0] / down_scale)
            print('Processing: {}'.format(IN_FILE))
            print('Width: {}, Height: {}'.format(nWidth, nHeight))

            assert nWidth == outJson['w'] and nHeight == outJson['h']

            # Add information for frame in new transforms metadata file
            frameInfo = dict()
            frameInfo['file_path'] = RGB_OUT_FILE

            # ORDERING HERE IS VERY IMPORTANT
            frameInfo['transform_matrix'] = tJSON['frames'][idx]['transform_matrix']

            # Process mask
            if use_masks:
                IN_FILE = '/'.join([mask_path, file])
                OUT_FILE = '/'.join([OUT_MASK_PATH, file])
                maskImage = iio.v3.imread(IN_FILE)
                nWidth, nHeight = int(maskImage.shape[1] / down_scale), int(maskImage.shape[0] / down_scale)
                print('Processing mask: {}'.format(IN_FILE))
                print('Width: {}, Height: {}'.format(nWidth, nHeight))

                assert nWidth == outJson['w'] and nHeight == outJson['h']
                maskImage = Image.fromarray(maskImage).resize((nWidth, nHeight))
                blackWhite = maskImage.convert('1')

                img[blackWhite == 0] = 255
                img = Image.fromarray(img).resize((nWidth, nHeight))
                iio.v3.imwrite(RGB_OUT_FILE, img)

                with open(OUT_FILE[:-4] + '.jpg', 'w') as maskImgFile:
                    blackWhite.save(maskImgFile)

                frameInfo['mask_path'] = OUT_FILE[:-4] + '.jpg'

            else:
                img = Image.fromarray(img).resize((nWidth, nHeight))
                iio.v3.imwrite(RGB_OUT_FILE, img)

            outJson['frames'].append(frameInfo)
            idx += 1


    # Write new transforms file
    with open('/'.join([out_path, 'transforms.json']), 'w') as tFile:
        json.dump(outJson, tFile, ensure_ascii=False, indent=4, cls=NumpyEncoder)

def own_to_masked(
        rgbPath: str,
        maskPath: str,
        outPath: str,
        extrFile: str,
        intrFile: str,
        useMask: bool = True,
        downScale: float = 1.0
):
    extrinsics = np.load(extrFile, allow_pickle=True)[()]
    intrinsics = np.load(intrFile, allow_pickle=True)[()]

    transforms = dict()

    # Assume no distortion
    transforms['k1'] = 0
    transforms['k2'] = 0
    transforms['p1'] = 0
    transforms['p2'] = 0

    # Original width, height
    oWidth, oHeight = 0,0

    # Infer the width/height of an image
    for key in extrinsics:
        RGB_FILE = '/'.join([rgbPath, key + '.png'])
        img = iio.v3.imread(RGB_FILE)
        oWidth, oHeight = img.shape[1], img.shape[0]
        break

    # Get intrinsics
    # As they are shared, use intrinsic of first camera
    for key in intrinsics:
        fx, fy, cx, cy = intrinsics[key][0][0], intrinsics[key][1][1], intrinsics[key][0][2], intrinsics[key][1][2]
        transforms['fl_x'] = float(fx/downScale)
        transforms['fl_y'] = float(fy/downScale)
        transforms['cx'] = float(cx/downScale)
        transforms['cy'] = float(cy/downScale)
        break

    # New width, height
    nWidth, nHeight = int(oWidth / downScale), int(oHeight / downScale)

    transforms['w'] = nWidth
    transforms['h'] = nHeight
    transforms['camera_model'] = 'OPENCV'

    transforms['frames'] = []

    OUT_RGB = '/'.join([outPath, 'images'])
    OUT_MASK = '/'.join([outPath, 'masks'])

    for dir in [OUT_RGB, OUT_MASK]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    for key in extrinsics:
        # Write mask + image
        IN_FILE = '/'.join([rgbPath, key + '.png'])
        OUT_FILE = '/'.join([OUT_RGB, key + '.png'])

        img = iio.v3.imread(IN_FILE)

        frameInfo = dict()
        frameInfo['file_path'] = OUT_FILE

        mat = np.array(extrinsics[key])

        # Extrinsics is here in OPENCV format
        # Convert to OPENGL
        extrinsics[key] = extrinsics[key].dot(CVtoGL)

        frameInfo['transform_matrix'] = np.array(extrinsics[key])

        # Process mask
        if useMask:
            IN_FILE_MASK = '/'.join([maskPath, key + '.png'])
            OUT_FILE_MASK = '/'.join([OUT_MASK, key + '.png'])
            maskImg = iio.v3.imread(IN_FILE_MASK)
            idx = (maskImg < 255)

            img = img.copy()
            img[idx] = 255
            img = Image.fromarray(img).resize((nWidth, nHeight))
            iio.v3.imwrite(OUT_FILE, img)

            maskImg = maskImg.copy()
            maskImg[maskImg < 255] = 0
            maskImg[maskImg > 0] = 255
            maskImg = Image.fromarray(maskImg).resize((nWidth, nHeight))
            iio.v3.imwrite(OUT_FILE_MASK[:-4] + '.jpg', maskImg)

            '''
            with open(OUT_FILE_MASK[:-4] + '.jpg', 'w') as maskImgFile:
                blackWhite.save(maskImgFile)
            '''

            frameInfo['mask_path'] = OUT_FILE_MASK[:-4] + '.jpg'
        else:
            img = Image.fromarray(img).resize((nWidth, nHeight))
            iio.v3.imwrite(OUT_FILE, img)

        transforms['frames'].append(frameInfo)

    with open('/'.join([outPath, 'transforms.json']), 'w') as tFile:
        json.dump(transforms, tFile, ensure_ascii=False, indent=4, cls=NumpyEncoder)


if __name__ == '__main__':
    '''
    TRANSFORM_PATH = '/mnt/hdd/data/head_38_nerfstudio'

    RGB_PATH = '/mnt/hdd/data/head_38/frame_00000/images'
    MASK_PATH = '/mnt/hdd/data/head_38/frame_00000/masks'

    OUT_PATH = '/mnt/hdd/data/head_38_own'

    nerfstudio_to_masked(
        RGB_PATH,
        TRANSFORM_PATH,
        OUT_PATH,
        MASK_PATH,
        True,
        down_scale=4
    )
    '''

    RGB_PATH = '/mnt/hdd/data/head_38/frame_00000/images'
    MASK_PATH = '/mnt/hdd/data/head_38/frame_00000/masks'

    OUT_PATH = '/mnt/hdd/data/head_38_own'
    EXTR_FILE = '/mnt/hdd/data/head_38/extrinsics.npy'
    INTR_FILE = '/mnt/hdd/data/head_38/intrinsics.npy'

    own_to_masked(
        RGB_PATH,
        MASK_PATH,
        OUT_PATH,
        EXTR_FILE,
        INTR_FILE,
        True,
        4.0
    )