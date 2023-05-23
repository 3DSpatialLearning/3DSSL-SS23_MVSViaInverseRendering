import os
import json
import re

import numpy as np
import imageio as iio
from PIL import Image

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
def main(
        rgb_path: str,
        transform_path: str,
        out_path: str,
        mask_path: str = '',
        use_masks: bool = False,
        down_scale: int = 1
):
    """
        Arguments:
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
    outJson['fl_x'] = tJSON['fl_x']
    outJson['fl_y'] = tJSON['fl_y']
    outJson['cx'] = int(tJSON['cx'] / down_scale)
    outJson['cy'] = int(tJSON['cy'] / down_scale)
    outJson['w'] = int(tJSON['w'] / down_scale)
    outJson['h'] = int(tJSON['h'] / down_scale)
    outJson['k1'] = tJSON['k1']
    outJson['k2'] = tJSON['k2']
    outJson['p1'] = tJSON['p1']
    outJson['p2'] = tJSON['p2']
    outJson['frames'] = []

    print('Output size: Width: {}, Height: {}'.format(outJson['w'], outJson['h']))

    for file in arr:
        # If file is image
        if file.endswith('.png') or file.endswith('.jpg'):

            IN_FILE = '/'.join([rgb_path, file])
            OUT_FILE = '/'.join([OUT_IMG_PATH, file])
            img = iio.v3.imread(IN_FILE)

            nWidth, nHeight = int(img.shape[1] / down_scale), int(img.shape[0] / down_scale)
            print('Processing: {}'.format(IN_FILE))
            print('Width: {}, Height: {}'.format(nWidth, nHeight))

            assert nWidth == outJson['w'] and nHeight == outJson['h']
            img = Image.fromarray(img).resize((nWidth, nHeight))
            iio.v3.imwrite(OUT_FILE, img)


            # Add information for frame in new transforms metadata file
            frameInfo = dict()
            frameInfo['file_path'] = OUT_FILE

            # ORDERING HERE IS VERY IMPORTANT
            frameInfo['transform_matrix'] = tJSON['frames'][idx]['transform_matrix']

            # Process mask
            if use_masks:
                IN_FILE = '/'.join([mask_path, file])
                OUT_FILE = '/'.join([OUT_MASK_PATH, file])
                img = iio.v3.imread(IN_FILE)
                nWidth, nHeight = int(img.shape[1] / down_scale), int(img.shape[0] / down_scale)
                print('Processing mask: {}'.format(IN_FILE))
                print('Width: {}, Height: {}'.format(nWidth, nHeight))

                assert nWidth == outJson['w'] and nHeight == outJson['h']
                img = Image.fromarray(img).resize((nWidth, nHeight))
                iio.v3.imwrite(OUT_FILE, img)

                frameInfo['mask_path'] = OUT_FILE

            outJson['frames'].append(frameInfo)
            idx += 1

    # Write new transforms file
    with open('/'.join([out_path, 'transforms.json']), 'w') as tFile:
        json.dump(outJson, tFile, ensure_ascii=False, indent=4, cls=NumpyEncoder)


if __name__ == '__main__':
    RGB_PATH = '/mnt/hdd/data/head_01'
    TRANSFORM_PATH = '/mnt/hdd/data/head_01_nerfstudio'

    MASK_PATH = '/mnt/hdd/data/head_01/masks'
    OUT_PATH = '/mnt/hdd/data/head_01_own'

    main(
        RGB_PATH,
        TRANSFORM_PATH,
        OUT_PATH,
        MASK_PATH,
        True,
        down_scale=2
    )