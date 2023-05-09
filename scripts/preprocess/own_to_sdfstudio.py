"""
    Script for converting own data format to format
    supported by sdfstudio
"""
import os
import json
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt

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

def own_to_sdfstudio(data_path: str, out_path: str,
                     frame: int, mask: bool) -> None:
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # General settings
    CAMERA = 'OPENCV'
    AABB_MIN = [-1, -1, -1]
    AABB_MAX = [1, 1 , 1]
    NEAR = 0.5
    FAR = 4.5

    # Scene box information
    scene_box = {
        'aabb': [AABB_MIN, AABB_MAX],
        'near': NEAR,
        'far': FAR,
        'radius': 1.0,
        'collider_type': 'near_far'
    }

    settings = dict()
    settings['camera_model'] = CAMERA
    settings['has_mono_prior'] = True
    settings['has_foreground_mask'] = mask
    settings['scene_box'] = scene_box
    settings['frames'] = []

    extrFile = '/'.join([data_path, 'extrinsics.npy'])
    intrFile = '/'.join([data_path, 'intrinsics.npy'])

    DEPTH_PATH = '/'.join([data_path, f'frame_{0:05}'.format(frame), 'depth'])
    IMG_PATH = '/'.join([data_path, f'frame_{0:05}'.format(frame), 'images'])
    MASK_PATH = '/'.join([data_path, f'frame_{0:05}'.format(frame), 'masks'])
    NORMAL_PATH = '/'.join([data_path, f'frame_{0:05}'.format(frame), 'normals'])

    cam2worlds = np.load(extrFile, allow_pickle=True)[()]
    intr = np.load(intrFile, allow_pickle=True)[()]

    dimWritten = False

    for i, key in enumerate(cam2worlds):
        cam2world_mat = cam2worlds[key]
        intr_mat = intr[key]

        # Add small offset
        # This was determined through visualization
        cam2world_mat[2,3] += 0.1

        # Settings per frame
        frame_settings = dict()
        frame_settings['camtoworld'] = cam2world_mat
        frame_settings['intrinsics'] = intr_mat
        frame_settings['rgb_path'] = '/'.join([IMG_PATH, key + '.png'])

        # Add image dimensions in first iteration
        if i == 0 or not dimWritten:
            if os.path.exists(frame_settings['rgb_path']):
                img = iio.v2.imread(frame_settings['rgb_path'])
                height = img.shape[0]
                width = img.shape[1]
                settings['height'] = height
                settings['width'] = width
                dimWritten = True
        frame_settings['mono_depth_path'] = '/'.join([DEPTH_PATH, key + '.npy'])
        frame_settings['mono_normal_path'] = '/'.join([NORMAL_PATH, key + '.npy'])
        if mask:
            frame_settings['foreground_mask'] = '/'.join([MASK_PATH, key + '.png'])
        settings['frames'].append(frame_settings)


    META_FILE = '/'.join([out_path, 'meta_data.json'])
    with open(META_FILE, 'w', encoding='utf-8') as jFile:
        json.dump(settings, jFile, ensure_ascii=False, indent=4, cls=NumpyEncoder)


if __name__ == '__main__':
    DATA_PATH = '../data/head_38'
    OUT_PATH = '../data/own_sdfstudio_head_38'
    own_to_sdfstudio(DATA_PATH, OUT_PATH, 0, True)