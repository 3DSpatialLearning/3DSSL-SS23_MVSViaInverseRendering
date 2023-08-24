import math

import numpy as np
import open3d as o3d

import point_cloud_utils as pcu
import trimesh
import trimesh as tri

import colorsys
import matplotlib

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

if __name__ == '__main__':
    '''
    mesh_distance_coloring(
        '/mnt/hdd/extracted/model_52.ply',
        '/mnt/hdd/data/mesh_85_f_150_prior.ply',
        '/mnt/hdd/metrics/model_52_color.ply'
    )
    '''