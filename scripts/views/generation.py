import torch
import numpy as np

# Translating point by t units in z direction
translation = lambda t : torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]
    ],
    device='cuda',
    dtype=torch.float32
)

rotation_phi = lambda phi : torch.tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]
    ],
    device='cuda',
    dtype=torch.float32
)

rotation_theta = lambda theta: torch.tensor(
    [
        [np.cos(theta), 0, -np.sin(theta), 0],
        [0, 1, 0, 0],
        [np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ],
    device='cuda',
    dtype=torch.float32
)

def cam2world_from_spherical(theta, phi, radius):
    """
        Given spherical coordinates (theta, phi, radius)
        return a cam2world matrix, representing a camera at that point
        looking at origin.

        Rotation is assumed to be in degree.
    """
    cam2world = translation(radius)
    cam2world = rotation_phi(phi / 180.0 * np.pi) @ cam2world
    cam2world = rotation_theta(theta / 180.0 * np.pi) @ cam2world
    '''
    cam2world = torch.tensor(
    [
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], device='cuda', dtype=torch.float32) @ cam2world
    '''
    return cam2world

# print(cam2world_from_spherical(-45.0, 128.0, 2.0))