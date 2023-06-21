from __future__ import annotations

import numpy as np
import scipy.stats as ss
from numpy.random import Generator
from tjax import NumpyRealArray, ShapeLike
from typing_extensions import Self, override

from .shaped_distribution import ShapedDistribution

__all__ = ['ScipyMultivariateNormal']


# pylint: disable=protected-access
mvn: type = ss._multivariate.multivariate_normal_frozen  # noqa: SLF001
class ScipyMultivariateNormalUnvectorized(mvn):
    """This class repairs multivariate_normal.

    See https://github.com/scipy/scipy/issues/7689.
    """
    @override
    def rvs(self,  # type: ignore[misc]
            size: ShapeLike | None = None,
            random_state: Generator | None = None) -> NumpyRealArray:
        retval = super().rvs(size=size, random_state=random_state)  # pyright: ignore
        if size is None:
            size = ()
        elif isinstance(size, int):
            size = (size,)
        else:
            size = tuple(size)
        return np.reshape(retval, size + self.mean.shape)


class ScipyMultivariateNormal(ShapedDistribution[ScipyMultivariateNormalUnvectorized]):
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
            cov = np.tile(np.eye(mean.shape[-1]), mean.shape[:-1] + (1, 1))
        shape = mean[..., -1].shape
        rvs_shape = (mean.shape[-1],)
        dtype = mean.dtype
        objects = np.empty(shape, dtype=ScipyMultivariateNormalUnvectorized)
        for i in np.ndindex(*shape):
            objects[i] = ScipyMultivariateNormalUnvectorized(mean[i], cov[i])
        return cls(shape, rvs_shape, dtype, objects)
