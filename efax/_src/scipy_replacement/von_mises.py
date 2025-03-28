from __future__ import annotations

import numpy as np
import scipy.stats as ss
from tjax import NumpyRealArray
from typing_extensions import override

from .shaped_distribution import ShapedDistribution


class ScipyVonMises(ShapedDistribution[object]):
    """This class allows distributions having a non-empty shape."""
    @override
    def __init__(self, kappa: NumpyRealArray, loc: NumpyRealArray | None = None) -> None:
        if loc is None:
            loc = np.zeros_like(kappa)
        shape = np.broadcast(kappa, loc).shape
        rvs_shape = ()
        dtype = np.result_type(kappa.dtype, loc.dtype)
        kappa = np.astype(np.broadcast_to(kappa, shape), dtype)
        loc = np.astype(np.broadcast_to(loc, shape), dtype)
        objects = np.empty(shape, dtype=np.object_)
        for i in np.ndindex(*shape):
            objects[i] = ss.vonmises(kappa[i], loc[i])
        super().__init__(shape, rvs_shape, dtype, objects)


class ScipyVonMisesFisher(ShapedDistribution[object]):
    """This class allows distributions having a non-empty shape."""
    @override
    def __init__(self, mu: NumpyRealArray, kappa: NumpyRealArray) -> None:
        assert mu.shape[:-1] == kappa.shape
        shape = kappa.shape
        rvs_shape = (mu.shape[-1],)
        dtype = np.result_type(kappa.dtype, mu.dtype)
        kappa = np.astype(kappa, dtype)
        mu = np.astype(mu, dtype)
        objects = np.empty(shape, dtype=np.object_)
        for i in np.ndindex(*shape):
            objects[i] = ss.vonmises_fisher(mu[i], kappa[i])
        super().__init__(shape, rvs_shape, dtype, objects)
