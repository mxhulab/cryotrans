__all__ = [
    'Wasserstein2Loss'
]

import cupy as cp
import torch
from typing import Dict, Tuple
from .utility import torch_to_cupy, cupy_to_torch, BLOCKDIM, BLOCKSIZE

ker_blur = cp.RawKernel(r'''
extern "C" __global__ void blur(
    const float* src,
    int size,
    int n,
    float B,
    float* dst)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size) {
        int offset = tid % n;
        int base = tid - offset;
        float w = src[tid];
        for (int i = 0; i < n; ++i)
            w = fmaxf(w, src[base + i] - (i - offset) * (i - offset) / B);
        float v = 0;
        for (int i = 0; i < n; ++i)
            v += expf(src[base + i] - (i - offset) * (i - offset) / B - w);
        dst[tid] = logf(v) + w;
    }
}
''', 'blur')

def _blur(b, x):
    tmp = cp.empty_like(x, dtype = cp.float32)
    for i in range(x.ndim):
        # move i-th axis to last.
        for j in range(i, x.ndim - 1): x = x.swapaxes(j, j + 1).copy()
        tmp = tmp.reshape(x.shape)

        # blur along last axis.
        ker_blur((BLOCKDIM(x.size), ), (BLOCKSIZE, ), (x, x.size, x.shape[-1], cp.float32(b), tmp))
        x, tmp = tmp, x

        # move last axis back to i.
        for j in range(x.ndim - 1, i, -1): x = x.swapaxes(j - 1, j).copy()
    return x

class Wasserstein2LossFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a0 : torch.Tensor, a1 : torch.Tensor, paras : Dict) -> torch.Tensor:
        assert torch.is_tensor(a0) and torch.is_tensor(a1)
        assert a0.shape == a1.shape
        assert a0.is_cuda and a1.is_cuda and a0.device == a1.device

        eps = paras.get('eps', 1e-3)
        delta = paras.get('delta', 1e-3)
        maxiter = paras.get('maxiter', 1000)
        batch = paras.get('batch', 100)
        pixel_size = 1 / max(a0.shape)

        a0 = torch_to_cupy(a0)
        a1 = torch_to_cupy(a1)
        with a0.device:
            b = eps / (pixel_size * pixel_size)
            la0 = cp.log(cp.fmax(a0, 1e-20))
            la1 = cp.log(cp.fmax(a1, 1e-20))
            phi0 = cp.zeros_like(a0, dtype = cp.float32)
            phi1 = cp.zeros_like(a1, dtype = cp.float32)
            norm_1 = lambda x : cp.sum(cp.abs(x)).item()
            na0 = norm_1(a0)

            for rd in range(maxiter):
                phi0 = la0 - _blur(b, phi1)
                phi1 = la1 - _blur(b, phi0)
                if rd == 0 or (rd + 1) % batch == 0:
                    err = norm_1(a0 - cp.exp(phi0 + _blur(b, phi1))) / na0
                    # print(f'  Round {rd + 1}, |i0 - P1|_1 / |i0|_1 = {err:.6f}.')
                    if err < delta: break

            loss = -eps * cp.sum(cp.exp(phi0 + _blur(b, phi1)))
            phi0 *= eps
            phi1 *= eps
            loss += cp.vdot(phi0, a0) + cp.vdot(phi1, a1)

        ctx.phi0 = cupy_to_torch(phi0)
        ctx.phi1 = cupy_to_torch(phi1)
        return cupy_to_torch(loss)

    @staticmethod
    def backward(ctx, grad_output : torch.Tensor) -> Tuple:
        return grad_output * ctx.phi0, grad_output * ctx.phi1, None

class Wasserstein2Loss(torch.nn.Module):

    def __init__(self, **paras):
        super().__init__()
        self.paras = paras

    def forward(self, a : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
        return Wasserstein2LossFunction.apply(a, b, self.paras)
