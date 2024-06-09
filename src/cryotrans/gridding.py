__all__ = [
    'Gridding'
]

import cupy as cp
import torch
from typing import Tuple
from .utility import torch_to_cupy, cupy_to_torch, BLOCKDIM, BLOCKSIZE

ker_gridding3d = cp.RawKernel(r'''
extern "C" __global__ void gridding3d(
    const float* rho,
    const float* xs,
    int m,
    float* vol,
    int n1,
    int n2,
    int n3,
    float pixel_size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m) {
        float x_ = (xs[tid * 3    ] - pixel_size / 2) / pixel_size;
        float y_ = (xs[tid * 3 + 1] - pixel_size / 2) / pixel_size;
        float z_ = (xs[tid * 3 + 2] - pixel_size / 2) / pixel_size;
        float wt = rho[tid];
        int x = floorf(x_);
        int y = floorf(y_);
        int z = floorf(z_);
        float dx = x_ - x;
        float dy = y_ - y;
        float dz = z_ - z;
        if (0 <= x     && x     < n1 && 0 <= y     && y     < n2 && 0 <= z     && z     < n3) atomicAdd(vol + ( x      * n2 + y    ) * n3 + z    , wt * (1 - dx) * (1 - dy) * (1 - dz));
        if (0 <= x + 1 && x + 1 < n1 && 0 <= y     && y     < n2 && 0 <= z     && z     < n3) atomicAdd(vol + ((x + 1) * n2 + y    ) * n3 + z    , wt *      dx  * (1 - dy) * (1 - dz));
        if (0 <= x     && x     < n1 && 0 <= y + 1 && y + 1 < n2 && 0 <= z     && z     < n3) atomicAdd(vol + ( x      * n2 + y + 1) * n3 + z    , wt * (1 - dx) *      dy  * (1 - dz));
        if (0 <= x + 1 && x + 1 < n1 && 0 <= y + 1 && y + 1 < n2 && 0 <= z     && z     < n3) atomicAdd(vol + ((x + 1) * n2 + y + 1) * n3 + z    , wt *      dx  *      dy  * (1 - dz));
        if (0 <= x     && x     < n1 && 0 <= y     && y     < n2 && 0 <= z + 1 && z + 1 < n3) atomicAdd(vol + ( x      * n2 + y    ) * n3 + z + 1, wt * (1 - dx) * (1 - dy) *      dz );
        if (0 <= x + 1 && x + 1 < n1 && 0 <= y     && y     < n2 && 0 <= z + 1 && z + 1 < n3) atomicAdd(vol + ((x + 1) * n2 + y    ) * n3 + z + 1, wt *      dx  * (1 - dy) *      dz );
        if (0 <= x     && x     < n1 && 0 <= y + 1 && y + 1 < n2 && 0 <= z + 1 && z + 1 < n3) atomicAdd(vol + ( x      * n2 + y + 1) * n3 + z + 1, wt * (1 - dx) *      dy  *      dz );
        if (0 <= x + 1 && x + 1 < n1 && 0 <= y + 1 && y + 1 < n2 && 0 <= z + 1 && z + 1 < n3) atomicAdd(vol + ((x + 1) * n2 + y + 1) * n3 + z + 1, wt *      dx  *      dy  *      dz );
    }
}
''', 'gridding3d')

ker_grad_gridding3d = cp.RawKernel(r'''
extern "C" __global__ void grad_gridding3d(
    const float* rho,
    const float* xs,
    int m,
    const float* dvol,
    int n1,
    int n2,
    int n3,
    float pixel_size,
    float* vs)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m) {
        float x_ = (xs[tid * 3    ] - pixel_size / 2) / pixel_size;
        float y_ = (xs[tid * 3 + 1] - pixel_size / 2) / pixel_size;
        float z_ = (xs[tid * 3 + 2] - pixel_size / 2) / pixel_size;
        float wt = rho[tid];
        int x = floorf(x_);
        int y = floorf(y_);
        int z = floorf(z_);
        float dx = x_ - x;
        float dy = y_ - y;
        float dz = z_ - z;
        float vx = 0;
        float vy = 0;
        float vz = 0;
        if (0 <= x     && x     < n1 && 0 <= y     && y     < n2 && 0 <= z     && z     < n3) {
            float grad = dvol[( x      * n2 + y    ) * n3 + z    ];
            vx += wt *    -  1  * (1 - dy) * (1 - dz) * grad;
            vy += wt * (1 - dx) *    -  1  * (1 - dz) * grad;
            vz += wt * (1 - dx) * (1 - dy) *    -  1  * grad;
        }
        if (0 <= x + 1 && x + 1 < n1 && 0 <= y     && y     < n2 && 0 <= z     && z     < n3) {
            float grad = dvol[((x + 1) * n2 + y    ) * n3 + z    ];
            vx += wt *       1  * (1 - dy) * (1 - dz) * grad;
            vy += wt *      dx  *    -  1  * (1 - dz) * grad;
            vz += wt *      dx  * (1 - dy) *    -  1  * grad;
        }
        if (0 <= x     && x     < n1 && 0 <= y + 1 && y + 1 < n2 && 0 <= z     && z     < n3) {
            float grad = dvol[( x      * n2 + y + 1) * n3 + z    ];
            vx += wt *    -  1  *      dy  * (1 - dz) * grad;
            vy += wt * (1 - dx) *       1  * (1 - dz) * grad;
            vz += wt * (1 - dx) *      dy  *    -  1  * grad;
        }
        if (0 <= x + 1 && x + 1 < n1 && 0 <= y + 1 && y + 1 < n2 && 0 <= z     && z     < n3) {
            float grad = dvol[((x + 1) * n2 + y + 1) * n3 + z    ];
            vx += wt *       1  *      dy  * (1 - dz) * grad;
            vy += wt *      dx  *       1  * (1 - dz) * grad;
            vz += wt *      dx  *      dy  *    -  1  * grad;
        }
        if (0 <= x     && x     < n1 && 0 <= y     && y     < n2 && 0 <= z + 1 && z + 1 < n3) {
            float grad = dvol[( x      * n2 + y    ) * n3 + z + 1];
            vx += wt *    -  1  * (1 - dy) *      dz  * grad;
            vy += wt * (1 - dx) *    -  1  *      dz  * grad;
            vz += wt * (1 - dx) * (1 - dy) *       1  * grad;
        }
        if (0 <= x + 1 && x + 1 < n1 && 0 <= y     && y     < n2 && 0 <= z + 1 && z + 1 < n3) {
            float grad = dvol[((x + 1) * n2 + y    ) * n3 + z + 1];
            vx += wt *       1  * (1 - dy) *      dz  * grad;
            vy += wt *      dx  *    -  1  *      dz  * grad;
            vz += wt *      dx  * (1 - dy) *       1  * grad;
        }
        if (0 <= x     && x     < n1 && 0 <= y + 1 && y + 1 < n2 && 0 <= z + 1 && z + 1 < n3) {
            float grad = dvol[( x      * n2 + y + 1) * n3 + z + 1];
            vx += wt *    -  1  *      dy  *      dz  * grad;
            vy += wt * (1 - dx) *       1  *      dz  * grad;
            vz += wt * (1 - dx) *      dy  *       1  * grad;
        }
        if (0 <= x + 1 && x + 1 < n1 && 0 <= y + 1 && y + 1 < n2 && 0 <= z + 1 && z + 1 < n3) {
            float grad = dvol[((x + 1) * n2 + y + 1) * n3 + z + 1];
            vx += wt *       1  *      dy  *      dz  * grad;
            vy += wt *      dx  *       1  *      dz  * grad;
            vz += wt *      dx  *      dy  *       1  * grad;
        }
        vs[tid * 3    ] = vx;
        vs[tid * 3 + 1] = vy;
        vs[tid * 3 + 2] = vz;
    }
}
''', 'grad_gridding3d')

class GriddingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rho : torch.Tensor, xs : torch.Tensor, shape : Tuple[int]) -> torch.Tensor:
        ctx.rho = rho
        ctx.xs = xs
        rho = torch_to_cupy(rho)
        xs = torch_to_cupy(xs)

        m = len(rho)
        pixel_size = 1 / max(shape)
        assert xs.shape == (m, 3) and rho.shape == (m, ) and len(shape) == 3
        assert rho.device == xs.device

        with rho.device:
            vol = cp.zeros(shape, dtype = cp.float32)
            ker_gridding3d((BLOCKDIM(m), ), (BLOCKSIZE, ), (rho, xs, m, vol, shape[0], shape[1], shape[2], cp.float32(pixel_size)))
            return cupy_to_torch(vol)

    @staticmethod
    def backward(ctx, grad_output : torch.Tensor) -> Tuple[torch.Tensor]:
        rho = torch_to_cupy(ctx.rho)
        xs = torch_to_cupy(ctx.xs)
        grad_output = torch_to_cupy(grad_output)

        shape = grad_output.shape
        m = len(rho)
        pixel_size = 1 / max(shape)
        assert xs.shape == (m, 3) and rho.shape == (m, ) and len(shape) == 3
        assert rho.device == xs.device == grad_output.device

        with rho.device:
            vs = cp.empty_like(xs, dtype = cp.float32)
            ker_grad_gridding3d((BLOCKDIM(m), ), (BLOCKSIZE, ), (rho, xs, m, grad_output, shape[0], shape[1], shape[2], cp.float32(pixel_size), vs))
            return None, cupy_to_torch(vs), None

class Gridding(torch.nn.Module):

    def __init__(self, shape : Tuple[int]):
        super().__init__()
        self.shape = shape

    def forward(self, rho : torch.Tensor, xs : torch.Tensor) -> torch.Tensor:
        return GriddingFunction.apply(rho, xs, self.shape)
