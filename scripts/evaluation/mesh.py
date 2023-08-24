import math

import numpy as np
import open3d as o3d

import point_cloud_utils as pcu
import trimesh
import trimesh as tri

import colorsys
import matplotlib

# Tolerance in Chmafer distance computation
# Here: 10mm
TOLERANCE = 10.0

def chamfer_box(pred, gt):
    """
        Compute chamfer distance inside bounding box of ground-truth point cloud
        with some tolerance. Avoids large distances due to outliers in prediction.
    """
    gt_x_min, gt_x_max = np.min(gt[:, 0]) - TOLERANCE, np.max(gt[:, 0]) + TOLERANCE
    gt_y_min, gt_y_max = np.min(gt[:, 1]) - TOLERANCE, np.max(gt[:, 1]) + TOLERANCE
    gt_z_min, gt_z_max = np.min(gt[:, 2]) - TOLERANCE, np.max(gt[:, 2]) + TOLERANCE
    mask_x = (pred[:, 0] >= gt_x_min) & (pred[:, 0] <= gt_x_max)
    mask_y = (pred[:, 1] >= gt_y_min) & (pred[:, 1] <= gt_y_max)
    mask_z = (pred[:, 2] >= gt_z_min) & (pred[:, 2] <= gt_z_max)
    mask = mask_x & mask_y & mask_z
    indices = np.where(mask)
    return pcu.chamfer_distance(pred[indices], gt)


def mesh_gt_chamfer_binary_search(
        pred_path: str,
        gt_path: str
):
    """
        Computes Chamfer distance (in mm) between the vertices of a predicted mesh
        and ground-truth mesh.

        Here, the scale of the predicted mesh is not known, the scale of ground-truth mesh
        is assumed to be in meters.

    """
    pred = o3d.io.read_triangle_mesh(pred_path)
    gt = o3d.io.read_triangle_mesh(gt_path)

    # Convert ground truth vertices to millimeter
    gt_vert = 100.0 * np.asarray(gt.vertices)
    pred_vert = np.asarray(pred.vertices)

    l_scale = 10E-1
    r_scale = 10E5

    eps = 10E-2
    steps = 20

    dist = 1E6
    scale = 0

    while steps > 0:
        scale = l_scale + (r_scale - l_scale) / 2

        dist = chamfer_box(scale * pred_vert, gt_vert)
        dist_left = chamfer_box((scale - eps) * pred_vert, gt_vert)
        dist_right = chamfer_box((scale + eps) * pred_vert, gt_vert)

        print(dist_left, dist, dist_right)

        if (dist_left > dist and dist_right > dist):
            return dist, scale
        elif (dist_left > dist and dist_right < dist):
            l_scale = scale
        else:
            r_scale = scale

        steps -= 1

    return dist, scale

def mesh_gt_chamfer(
        pred_path: str,
        gt_path: str
):
    """
        Returns the Chamfer distance between the vertices of a predicted mesh
        and an assumed ground-truth mesh.

        Both meshes are assumed to be in meters.

        pred_path: path to the predicted mesh
        gt_path: path to the ground-truth mesh
    """
    pred = o3d.io.read_triangle_mesh(pred_path)
    gt = o3d.io.read_triangle_mesh(gt_path)

    # Convert to millimeters
    pred_vert = 100.0 * np.asarray(pred.vertices)
    gt_vert = 100.0 * np.asarray(gt.vertices)

    return pcu.chamfer_distance(pred_vert, gt_vert)


def mesh_distance_coloring(
        mPredPath: str,
        mGTPath: str,
        mOutPath: str
):
    """
        Given two input meshes, output a colored mesh that is colored according
        to distance of nearest neighbor
    """

    predMesh = o3d.io.read_triangle_mesh(mPredPath)
    gtMesh = o3d.io.read_triangle_mesh(mGTPath)

    # Convert to millimeters
    predVertices = 100 * np.asarray(predMesh.vertices)
    face1 = np.asarray(predMesh.triangles)


    gtVertices = 100 * np.asarray(gtMesh.vertices)
    face2 = np.asarray(gtMesh.triangles)

    print('Chamfer distance: {}'.format(pcu.chamfer_distance(predVertices, gtVertices)))


    sdf, _, _ = pcu.signed_distance_to_mesh(predVertices, gtVertices, face2)
    sdf = np.sqrt(np.abs(sdf))
    sdf_min, sdf_max = np.min(sdf), np.max(sdf)
    sdf = (sdf - sdf_min) / (sdf_max - sdf_min)

    print('Min x: {}, Max x: {}'.format(np.min(predVertices[:,0]), np.max(predVertices[:,0])))
    print('Min y: {}, Max y: {}'.format(np.min(predVertices[:,1]), np.max(predVertices[:,1])))
    print('Min z: {}, Max z: {}'.format(np.min(predVertices[:,2]), np.max(predVertices[:,2])))

    '''
    # Assign sdf to bins
    NUM_BINS = 64

    bins = np.linspace(0, 1, NUM_BINS)
    sdf_digit = np.digitize(sdf, bins)

    discr = np.zeros(sdf_digit.shape)

    for i in range(NUM_BINS):
        discr[sdf_digit == i] = bins[i]
    '''

    assert sdf.shape[0] == predVertices.shape[0], "Not same shape"

    # print('Unique values: {}'.format(np.unique(discr).shape[0]))

    # Squared distance
    # sdf = np.abs(sdf)

    colors = np.zeros((sdf.shape[0], 3))
    colormap = matplotlib.cm.get_cmap('turbo')
    map_colors = colormap(sdf)
    colors = map_colors[:,:3]

    coloredMesh = trimesh.Trimesh(predVertices / 100.0, face1, vertex_colors=colors)
    coloredMesh.export(mOutPath)


def evaluation():
    """
        Run all evaluations for the project
    """

    # Colmap evaluation

    # Head 85 - Frame 150; Downscaling factor 2; White background
    # head_85_mesh = '/mnt/hdd/colmap/dense/0/fused.ply'
    # head_85_gt = '/mnt/hdd/data/mesh_85_f_150_gt.ply'

    # Chamfer distance in millimeter
    # chamfer, scale = mesh_gt_chamfer_binary_search(head_85_mesh, head_85_gt)
    # print(chamfer, scale)
    pass

def eval_model_005():
    head_85_mesh = '/mnt/hdd/evaluation/model_eval_005/mesh.ply/poisson_mesh.ply'
    head_85_gt = '/mnt/hdd/data/mesh_85_f_150_gt.ply'
    c = mesh_gt_chamfer(head_85_mesh, head_85_gt)
    print('Model Eval 005 | 20k iterations == Chamfer: {}'.format(c))


def eval_model_006():
    head_85_mesh = '/mnt/hdd/evaluation/model_eval_006/mesh.ply'
    head_85_gt = '/mnt/hdd/data/mesh_85_f_150_gt.ply'

    c = mesh_gt_chamfer(head_85_mesh, head_85_gt)
    print('Model Eval 006 | 10k iterations == Chamfer: {}'.format(c))

def eval_model_007():
    head_85_mesh = '/mnt/hdd/evaluation/model_eval_007/mesh.ply'
    head_85_gt = '/mnt/hdd/data/mesh_85_f_150_gt.ply'

    c = mesh_gt_chamfer(head_85_mesh, head_85_gt)
    print('Model Eval 007 | 10k iterations == Chamfer: {}'.format(c))

def eval_model_008():
    head_85_mesh = '/mnt/hdd/evaluation/model_eval_008/mesh.ply'
    head_85_gt = '/mnt/hdd/data/mesh_85_f_150_gt.ply'

    c = mesh_gt_chamfer(head_85_mesh, head_85_gt)
    print('Model Eval 008 | 10k iterations == Chamfer: {}'.format(c))

def eval_model_009():
    head_85_mesh = '/mnt/hdd/evaluation/model_eval_009/mesh.ply'
    head_85_gt = '/mnt/hdd/data/mesh_85_f_150_gt.ply'

    c = mesh_gt_chamfer(head_85_mesh, head_85_gt)
    print('Model Eval 009 | 10k iterations == Chamfer: {}'.format(c))

def eval_model_010():
    head_85_mesh = '/mnt/hdd/evaluation/model_eval_010/mesh.ply'
    head_85_gt = '/mnt/hdd/data/mesh_85_f_150_gt.ply'

    c = mesh_gt_chamfer(head_85_mesh, head_85_gt)
    print('Model Eval 010 | 10k iterations == Chamfer: {}'.format(c))


if __name__ == '__main__':
    '''
    mesh_distance_coloring(
        '/mnt/hdd/extracted/model_52.ply',
        '/mnt/hdd/data/mesh_85_f_150_prior.ply',
        '/mnt/hdd/metrics/model_52_color.ply'
    )
    '''

    # evaluation()
    eval_model_005()
    print()

    # Experimentation on Eikonal loss
    eval_model_006()
    eval_model_007()
    eval_model_008()
    eval_model_009()
    eval_model_010()