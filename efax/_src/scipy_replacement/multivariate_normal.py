from typing import Optional

import numpy as np
from numpy.random import Generator
from scipy import stats as ss
from tjax import RealArray, ShapeLike

from .shaped_distribution import ShapedDistribution

__all__ = ['ScipyMultivariateNormal']


# pylint: disable=protected-access
class ScipyMultivariateNormalFixRVs(ss._multivariate.multivariate_normal_frozen):
    """
    This class repairs multivariate_normal.  See https://github.com/scipy/scipy/issues/7689.
    """
    def rvs(self,
            size: Optional[ShapeLike] = None,
            random_state: Optional[Generator] = None) -> RealArray:
        retval = super().rvs(size=size, random_state=random_state)
        if size is None:
            size = ()
        elif isinstance(size, int):
            size = (size,)
        else:
            size = tuple(size)
        return retval.reshape(size + self.mean.shape)


class ScipyMultivariateNormal(ShapedDistribution):
    """
    This class allows distributions having a non-empty shape.
    """
    def __init__(self, mean: Optional[RealArray] = None, cov: Optional[RealArray] = None):
        if mean is None and cov is None:
            mean = np.zeros(1)
        if mean is None:
            if cov is None:
                raise ValueError
            mean = np.zeros(cov.shape[:-1], dtype=cov.dtype)
        if cov is None:
            cov = np.tile(np.eye(mean.shape[-1]), mean.shape[:-1] + (1, 1))  # type: ignore
        shape = mean[..., -1].shape
        component_shape = (mean.shape[-1],)
        dtype = mean.dtype
        objects = np.empty(shape, dtype=np.object_)
        for i in np.ndindex(shape):
            objects[i] = ScipyMultivariateNormalFixRVs(mean[i], cov[i])
        super().__init__(shape, component_shape, dtype, objects)
