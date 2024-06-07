__all__ = [
    'ODENet'
]

import torch
from .velocity_net import VelocityNet

class ODENet(torch.nn.Module):

    def __init__(self, n_frames : int = 10, **vnet_kwargs):
        super().__init__()
        self.n_frames = n_frames
        self.v = VelocityNet(**vnet_kwargs)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_frames):
            dx = self.v(x) / self.n_frames
            x = x + dx
        return x

    def trajectory(self, x : torch.Tensor):
        yield x
        for _ in range(self.n_frames):
            dx = self.v(x) / self.n_frames
            x = x + dx
            yield x
