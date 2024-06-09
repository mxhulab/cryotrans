__all__ = [
    'VelocityNet'
]

import torch

class VelocityNet(torch.nn.Module):

    def __init__(self, depth : int = 3, width : int = 100, dim : int = 3):
        super().__init__()
        self.depth = depth
        for i in range(depth):
            layer = torch.nn.Linear(dim if i == 0 else width, dim if i == depth - 1 else width)
            torch.nn.init.normal_(layer.weight, std = 0.01)
            torch.nn.init.zeros_(layer.bias)
            self.__setattr__(f'n{i}', layer)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for i in range(self.depth):
            x = self.__getattr__(f'n{i}')(x)
            if i < self.depth - 1:
                x = torch.nn.functional.leaky_relu(x)
        return x
