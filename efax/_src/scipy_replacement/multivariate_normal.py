from __future__ import annotations

from typing import Self

import numpy as np
import optype.numpy as onp
import scipy.stats as ss
from numpy.random import Generator
from tjax import NumpyRealArray, Shape

from .shaped_distribution import ShapedDistribution


class ScipyMultivariateNormalUnvectorized:
    """This class repairs multivariate_normal.

    See https://github.com/scipy/scipy/issues/7689.
    """
    def __init__(self, mean: NumpyRealArray, cov: NumpyRealArray) -> None:
        super().__init__()
        self.distribution = ss.multivariate_normal(mean=mean, cov=cov)

    def pdf(self, x: NumpyRealArray) -> NumpyRealArray:
        return np.asarray(self.distribution.pdf(x))

    def sample(self, shape: Shape = (), *, rng: Generator | None = None) -> onp.ArrayND[np.float64]:
        retval = self.distribution.rvs(size=shape, random_state=rng)
        return np.reshape(retval, shape + self.distribution.mean.shape)

    def entropy(self) -> NumpyRealArray:
        return np.asarray(self.distribution.entropy())


class ScipyMultivariateNormal(
        ShapedDistribution[ScipyMultivariateNormalUnvectorized]):  # type: ignore # pyright: ignore
    """This class allows distributions having a non-empty shape."""
    @classmethod
    def from_mc(cls,
                mean: NumpyRealArray | None = None,
                cov: NumpyRealArray | None = None) -> Self:
        if mean is None and cov is None:
            mean = np.zeros(1)
        if mean is None:
            if cov is None:
                raise ValueError
            mean = np.zeros(cov.shape[:-1], dtype=cov.dtype)
        if cov is None:
            cov = np.tile(np.eye(mean.shape[-1]), (*mean.shape[:-1], 1, 1))
        shape = mean[..., -1].shape
        rvs_shape = (mean.shape[-1],)
        dtype = mean.dtype
        objects = np.empty(shape, dtype=ScipyMultivariateNormalUnvectorized)
        for i in np.ndindex(*shape):
            objects[i] = ScipyMultivariateNormalUnvectorized(mean[i], cov[i])
        return cls(shape, rvs_shape, dtype, objects, multivariate=True)
