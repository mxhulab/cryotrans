import cupy as cp
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack

BLOCKSIZE = 1024
BLOCKDIM = lambda x : (x - 1) // BLOCKSIZE + 1

def cupy_to_torch(x : cp.ndarray) -> torch.Tensor:
    return from_dlpack(x.toDlpack())

def torch_to_cupy(x : torch.Tensor) -> cp.ndarray:
    return cp.fromDlpack(to_dlpack(x))
