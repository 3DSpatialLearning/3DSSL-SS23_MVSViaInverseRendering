import torch
import numpy as np
import json

from PIL import Image

def get_color_gt(points):
    """
        For a given set of points in world coordinates, compute their vertex color
        by back-projecting them into image space.
    """
    CONFIG_FILE = '/mnt/hdd/data/head_85_f_150_d_2/meta_data.json'
    colors = []
    points_hom = np.concatenate([points, np.ones((points.shape[0], 1))], 1)

    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
        IMG_WIDTH = int(config['width'])
        IMG_HEIGHT = int(config['height'])
        for idx, frame in enumerate(config['frames']):
            K = np.array(frame['intrinsics']).astype(np.float32)[:3, :3]
            c2w = np.array(frame['camtoworld']).astype(np.float32)
            c2w[0:3, 1:3] *= -1

            # Convert to 4x4 so invertible
            # c2w = np.concatenate([c2w, np.array([0,0,0,1]).reshape(1,4)], 0)

            # 3x4 world_to_cam matrix
            w2c = np.linalg.inv(c2w)[:3]

            # Points in camera coordinate system
            points_cam = w2c @ points_hom.T
            points_cam[1:] *= -1

            # Points in image coordinate system
            # NUM POINTS x 3
            points_image = (K @ points_cam).T

            depth = (points_image[:, -1] + 1E-7).reshape(-1, 1)

            # Divide by depth to get points on image plane
            points_img_plane = (points_image[:, :2] / depth).astype(np.float32)

            # Clip values to get only pixels in valid range
            points_img_plane[:, 0] = np.clip(points_img_plane[:, 0], 0, IMG_WIDTH - 1)
            points_img_plane[:, 1] = np.clip(points_img_plane[:, 1], 0, IMG_HEIGHT - 1)

            # COLOR PART
            image = Image.open(frame['rgb_path']).convert('RGB')
            image = np.array(image)

            maskImg = Image.open(frame['foreground_mask']).convert('1')  # Black and white mask
            mask = np.array(maskImg)

            # Store image with projected points
            if idx == 0:
                rnd_indices = np.random.choice(points_img_plane.shape[0], 216)
                x_coords = np.floor(points_img_plane[rnd_indices, 0]).astype(np.uint16)
                y_coords = np.floor(points_img_plane[rnd_indices, 1]).astype(np.uint16)

                img_out = image.copy()
                img_out[y_coords, x_coords, 0] = 255
                img_out[y_coords, x_coords, 1] = 0
                img_out[y_coords, x_coords, 2] = 0

                im_out = Image.fromarray(img_out)
                im_out.save('/mnt/hdd/data/out.png')

            # Color for individual image
            image = image.astype(np.float32)

            # Set to value outside of image to not interpolate
            image[mask == False, :] = np.nan

            # i_color = []

            x_coords = np.floor(points_img_plane[:, 0]).astype(np.uint16)
            y_coords = np.floor(points_img_plane[:, 1]).astype(np.uint16)

            i_color = image[y_coords, x_coords, :]
            '''
            remap_chunk = int(3e4)
            for i in range(0, points_img_plane.shape[0], remap_chunk):
                R_interp = [cv2.remap(image[:,0],
                                     points_img_plane[i:i + remap_chunk, 0],
                                     points_img_plane[i:i + remap_chunk, 1],
                                     interpolation=cv2.INTER_LINEAR)[:, 0]]
                G_interp = [cv2.remap(image[:, 1],
                                      points_img_plane[i:i + remap_chunk, 0],
                                      points_img_plane[i:i + remap_chunk, 1],
                                      interpolation=cv2.INTER_LINEAR)[:, 0]]
                B_interp = [cv2.remap(image[:, 2],
                                  points_img_plane[i:i + remap_chunk, 0],
                                  points_img_plane[i:i + remap_chunk, 1],
                                  interpolation=cv2.INTER_LINEAR)[:, 0]]
            i_color = np.vstack(np.hstack([R_interp, G_interp, B_interp]))
            '''

            # Combine color
            colors.append(i_color.reshape(-1, 3, 1))
            break

        comb_colors = np.dstack(colors)
        R = np.nanmean(comb_colors[:, 0, :], axis=-1).astype(np.uint8).reshape(-1, 1)
        G = np.nanmean(comb_colors[:, 1, :], axis=-1).astype(np.uint8).reshape(-1, 1)
        B = np.nanmean(comb_colors[:, 2, :], axis=-1).astype(np.uint8).reshape(-1, 1)
        A = 255 * np.ones((R.shape[0], 1)).astype(np.uint8)
        COLORS = np.hstack([R, G, B, A])
        return COLORS


def get_colors(points, normals, pipeline):
    """
        Compute vertex color by querying a color network
    """
    field = pipeline.model.field

    points = torch.from_numpy(points).to('cuda').type(torch.float32)

    normals = torch.from_numpy(normals.copy()).to('cuda').type(torch.float32)
    points.requires_grad_(True)

    CONFIG_FILE = '/mnt/hdd/data/head_85_f_150_d_2/meta_data.json'

    colors = []

    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
        IMG_WIDTH = int(config['width'])
        IMG_HEIGHT = int(config['height'])
        for idx, frame in enumerate(config['frames']):
            K = np.array(frame['intrinsics']).astype(np.float32)[:3, :3]
            c2w = np.array(frame['camtoworld']).astype(np.float32)
            c2w[0:3, 1:3] *= -1

            # Position of camera
            cam_pos = torch.tensor(c2w[:3, 3], dtype=torch.float32).to('cuda')
            directions = torch.nn.functional.normalize(points, p=2, dim=-1)
            # directions = torch.zeros(points.shape).to('cuda')
            normals = torch.nn.functional.normalize(normals, p=2, dim=-1)

            with torch.enable_grad():
                h = field.forward_geonetwork(points)
                sdf, geo_feature = torch.split(h, [1, field.config.geo_feat_dim], dim=-1)

            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf, inputs=points, grad_outputs=d_output, create_graph=True, retain_graph=True,
                only_inputs=True
            )[0]
            field.training = True
            rgb = field.get_colors(points, directions, gradients, geo_feature, None)

            colors = rgb.detach().cpu().numpy()
            colors = 255 * colors
            break

        comb_colors = np.dstack(colors)
        R = np.mean(comb_colors[:, 0, :], axis=-1).astype(np.uint8).reshape(-1, 1)
        G = np.mean(comb_colors[:, 1, :], axis=-1).astype(np.uint8).reshape(-1, 1)
        B = np.mean(comb_colors[:, 2, :], axis=-1).astype(np.uint8).reshape(-1, 1)
        A = 255 * np.ones((R.shape[0], 1)).astype(np.uint8)
        COLORS = np.hstack([R, G, B, A])
        return COLORS