__all__ = [
    'particles'
]

import numpy as np
from typing import Tuple
from numpy.typing import NDArray

def particles(vol : NDArray[np.float32], threshold : float = 1e-6) -> Tuple[NDArray[np.float32]]:
    n = vol.shape[0]
    vol = np.where(vol < threshold, 0, vol)
    xs = np.nonzero(vol)
    rho = vol[xs]
    xs = np.vstack(xs)
    xs = (xs.transpose().copy() + 0.5) / n
    return rho, xs
