import json
import numpy as np
import pyvista as pv
import numpy.linalg as linalg

from scipy.spatial.transform import Rotation as R
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.pyvista import add_floor, add_coordinate_axes, \
                            add_camera_frustum, add_coordinate_system, \
                            Pose, Intrinsics

def read_transformation_sdfstudio(tPath: str, fName: str):
    """
    Read in transformation information from meta_data.json file used by sdfstudio.

    Parameters
    ----------
        tPath:
            Path to meta_data.json

    Returns
    -------
        Tuple (e,i) continaing extrinsics (e) and intrinsics (i)
    """
    tFile = '/'.join([tPath, fName])
    with open(tFile, 'r') as f:
        data = json.load(f)

        extrinsics = []
        intrinsics = []

        for element in data['frames']:
            cam2world = np.array(element['camtoworld'])
            intr = np.array(element['intrinsics'])
            extrinsics.append(cam2world)

            # In case intrinsics is a 4x4 matrix, trim down to 3x3
            if (intr.shape == (4,4)):
                intr = intr[:3,:3]

            intrinsics.append(intr)

    return (extrinsics, intrinsics)


def read_transformation_nerfstudio(tPath: str):
    """
    Read in transformation information from a transforms.json file generated/used by nerfstudio.
    

    Parameters
    ----------
        tPath:
            Path to the transforms.json file

    Returns
    -------
        Tuple (e,i) containing extrinsics (e) and intrinsics (i)
    """
    tFile = '/'.join([tPath, 'transforms.json'])

    with open(tFile, 'r') as f:
        data = json.load(f)
        # Intrinsic parameters
        cx : float = data['cx']
        cy : float = data['cy']
        fx : float = data['fl_x']
        fy : float = data['fl_y']

        # Build matrix
        intr_mat = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        extrinsics : list = []
        for element in data['frames']:
            cam2world = element['transform_matrix']
            extrinsics.append(np.array(cam2world))
        return (extrinsics, [intr_mat])
    

def read_transformation_custom_data(path: str) :
    """
    Read extrinsic and intrinsic information from custom dataset, i.e.
    assumes that .npy files are present. 

    Parameters
    ----------
        path:
            Path to the folder containing the extrinsics/intrinsics.npy files

    Returns
    -------
        Tuple (e,i) containing extrinsics (e) and intrinsics (i)
    """


    extr_file = '/'.join([path, 'extrinsics.npy'])
    intr_file = '/'.join([path, 'intrinsics.npy'])

    c2w_matrices = np.load(extr_file, allow_pickle=True)[()]
    intr_matrices = np.load(intr_file, allow_pickle=True)[()]

    extr_list = []
    intr_list = []

    for key in c2w_matrices:
        extr_list.append(c2w_matrices[key])
        intr_list.append(intr_matrices[key])

    return (extr_list, intr_list)


def plot(extrinsics, intrinsics, near: float = 1.0, far: float = 10.0, camConvention=CameraCoordinateConvention.OPEN_CV) -> None:
    """
    Plot a list of extrinsic and intrinsic camera parameters.

    Parameters
    ----------
        camConvention:
            The camera convention used in the extrinsics matrix.
    """
    
    p = pv.Plotter(notebook=False)

    # Floor is (x,z) plane
    add_floor(p, square_size=1.0, max_distance=10, axes=(1,2))
    add_coordinate_axes(p, scale=1.0, draw_labels=True)

    for i, cam2world in enumerate(extrinsics):
        # If only one intrinsics parameter, assume it's shared
        if len(intrinsics) == 1:
            pv_intr = Intrinsics(intrinsics[0])
        else:
            pv_intr = Intrinsics(intrinsics[i])


        pCamera = np.array([0,0,0,1])
        pCameraDir = np.array([0,0,1,1])
        
        pCamWorld = cam2world.dot(pCamera)
        pCamDirWorld = cam2world.dot(pCameraDir)

        norm = linalg.norm(pCamDirWorld)
        pCamDirWorld /= norm

        pNear = pCamWorld - near * pCamDirWorld
        pFar = pCamWorld - far * pCamDirWorld

        points = np.stack([pNear[:3], pFar[:3]])
        
        p.add_points(points, render_points_as_spheres=True, point_size=16.0)

        pv_cam2world = Pose(cam2world, pose_type=PoseType.CAM_2_WORLD,
                            camera_coordinate_convention=camConvention)
        add_camera_frustum(p, pv_cam2world, pv_intr, size=0.4, line_width=2, look_vector_length=4.5)
        add_coordinate_system(p, pv_cam2world, scale=0.4)
    p.show()


if __name__ == '__main__':
    # extr, intr = read_transformation_custom_data('../data/head_38/own')
    
    # Use transforms.json
    # nerfstudio transforms.json uses OPEN_GL camera convention
    # VERY CONFUSING as camera model is specified as OPEN_CV

    # Extrinsics/Intrinsics in sdfstudio format when running own pipeline
    extr, intr = read_transformation_sdfstudio('../data/head_01', 'meta_data_old.json')
    
    for ext in extr:
        ext[0,3] += 0.5
        ext[1,3] += 0.5
    

    plot(extr, intr, 1.0, 5.0, CameraCoordinateConvention.OPEN_CV)

