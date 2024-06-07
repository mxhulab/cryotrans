__all__ = [
    'tetramesh',
    'MeshLoss'
]

import numpy as np
import torch
from numpy.typing import NDArray

def tetramesh(xs : NDArray[np.float32], n : NDArray[np.float32]) -> NDArray[np.float32]:
    vertices = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]) / n
    tetras = vertices[np.array([
        [0, 4, 6, 7],
        [0, 4, 5, 7],
        [0, 1, 5, 7],
        [0, 1, 3, 7],
        [0, 2, 3, 7],
        [0, 2, 6, 7]
    ], dtype = np.int32)]
    return np.reshape(xs[:, None, None] + tetras, (-1, 4, 3))

class MeshLoss(torch.nn.Module):

    def __init__(self, mesh : torch.Tensor):
        super().__init__()
        self.n = len(mesh)
        self.mesh = mesh
        self.inv = torch.linalg.inv(mesh[..., 1:4, :] - mesh[..., 0:1, :])

    def forward(self, odenet : torch.nn.Module) -> torch.Tensor:
        mesh_out = odenet(self.mesh)
        afftrans = self.inv @ (mesh_out[..., 1:4, :] - mesh_out[..., 0:1, :])
        svdvals = torch.linalg.svdvals(afftrans)
        return torch.sum(torch.square(svdvals ** 2 - 1)) / self.n
