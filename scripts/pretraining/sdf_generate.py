from typing import Tuple, Union

import open3d as o3d
import numpy as np
import os

import trimesh as tri
import point_cloud_utils as pcu
from numpy import ndarray

def sampleBarycentric(v1: np.ndarray,
                      v2: np.ndarray,
                      v3: np.ndarray, numSamples: int) -> np.ndarray:
    r1 = np.random.uniform(0.0, 1.0, (numSamples, 1))
    r2 = np.random.uniform(0.0, 1.0, (numSamples, 1))
    c1 = np.dot((1 - np.sqrt(r1)), v1.reshape(1, 3))
    c2 = np.dot(np.sqrt(r1) * (1.0 - r2), v2.reshape(1, 3))
    c3 = np.dot(np.sqrt(r1) * r2, v3.reshape(1, 3))
    return c1 + c2 + c3

def generateSurfaceSamples(
        vertices: np.ndarray,
        indices: np.ndarray,
        samplesPerTriangle: int,
        includeCorner: bool = False,
        includeNormals: bool = False
    ):
    assert indices.shape[1] == 3
    assert vertices.shape[1] == 3

    tSize = samplesPerTriangle + 3 if includeCorner else samplesPerTriangle
    total = np.zeros((indices.shape[0] * tSize, 3))

    if includeNormals:
        normals = np.zeros((indices.shape[0] * tSize, 3))

    for i_idx, idx in enumerate(indices):
        f, s, t = idx
        p0, p1, p2 = vertices[f], vertices[s], vertices[t]

        # Calculate normal
        v1 = vertices[s] - vertices[f]
        v2 = vertices[t] - vertices[f]
        n = np.cross(v1, v2)
        n = n / np.sqrt(np.sum(n**2))

        # Sample points uniformly on triangle
        samples = sampleBarycentric(p0, p1, p2, samplesPerTriangle)
        eps = np.random.normal(0.0, 1.0, (samplesPerTriangle, 1))
        if includeNormals:
            normals[i_idx * tSize : (i_idx + 1) * tSize] = np.tile(n, tSize).reshape(-1, 3)
        total[i_idx * tSize : (i_idx + 1) * tSize] = samples + 0.01 * np.dot(eps, n.reshape(1, 3))

    if includeNormals:
        ret = (total, normals)
    else:
        ret = total,

    return ret


def generateSyntheticSDFData(
        meshtGTPath: str,
        outPath: str,
        samplesPerTriangle: int,
        includeNormals: bool = False
    ):
    """
        meshGTPath: path to ground truth mesh
    """
    mesh = o3d.io.read_triangle_mesh(meshtGTPath)
    vertices = np.asarray(mesh.vertices)
    indices = np.asarray(mesh.triangles)

    if includeNormals:
        surface_samples, normals = generateSurfaceSamples(vertices, indices,
                                                          samplesPerTriangle,
                                                          includeCorner=False,
                                                          includeNormals=includeNormals)
    else:
        surface_samples = generateSurfaceSamples(vertices, indices, samplesPerTriangle, False)

    # Number of samples in box
    # Default: 25% of surface samples
    num_box_samples = int(0.25 * surface_samples.shape[0])

    # Scale box in which samples are drawn
    SCALE = 2
    x_min, x_max = SCALE * vertices[:,0].min(), SCALE * vertices[:,0].max()
    y_min, y_max = SCALE * vertices[:,1].min(), SCALE * vertices[:,1].max()
    z_min, z_max = SCALE * vertices[:,2].min(), SCALE * vertices[:,2].max()

    # Uniform sampling
    box_samples = np.zeros((num_box_samples,3))
    box_samples[:, 0] = np.random.uniform(x_min, x_max, num_box_samples)
    box_samples[:, 1] = np.random.uniform(y_min, y_max, num_box_samples)
    box_samples[:, 2] = np.random.uniform(z_min, z_max, num_box_samples)


    # Normals here are 0 => can be later used for masking
    if includeNormals:
        box_normals = np.zeros((num_box_samples, 3))
        normals = np.vstack((normals, box_normals))

    # Stack samples together
    combined_samples = np.vstack((surface_samples, box_samples))

    print('Total number of samples: {}'.format(combined_samples.shape))

    # Compute signed distance
    sdf, _, _ = pcu.signed_distance_to_mesh(combined_samples, vertices, indices)

    # Combine points + sdf
    out = np.hstack((combined_samples, sdf.reshape(-1, 1)))

    outDict = dict()

    assert combined_samples.shape[0] == sdf.shape[0]

    outDict['samples'] = combined_samples
    outDict['sdf'] = sdf

    if includeNormals:
        assert normals.shape[0] == combined_samples.shape[0]
        outDict['normals'] = normals

    if not os.path.exists(outPath):
        os.makedirs(outPath)

    np.save('/'.join([outPath, 'samples_normals.npy']), outDict, allow_pickle=True)


if __name__ == '__main__':
    generateSyntheticSDFData(
        '/mnt/hdd/data/mesh_85_f_150_prior.ply',
        '/mnt/hdd/data/sdf_prior/',
        samplesPerTriangle = 6,
        includeNormals=True
    )