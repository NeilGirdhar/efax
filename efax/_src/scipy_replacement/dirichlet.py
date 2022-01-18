from typing import Optional

import numpy as np
import scipy.stats as ss
from numpy.random import Generator
from tjax import ComplexArray, RealArray, ShapeLike

from .shaped_distribution import ShapedDistribution

__all__ = ['ScipyDirichlet']


# pylint: disable=protected-access
class ScipyDirichletFixRVsAndPDF(ss._multivariate.dirichlet_frozen):
    """
    This class repairs dirichlet.  See https://github.com/scipy/scipy/issues/6005 and
    https://github.com/scipy/scipy/issues/6006.
    """
    def rvs(self,
            size: Optional[ShapeLike] = None,
            random_state: Optional[Generator] = None) -> RealArray:
        if size is None:
            size = ()
        # This somehow fixes the behaviour of rvs.
        return super().rvs(size=size, random_state=random_state)

    def pdf(self, x: RealArray) -> RealArray:
        if x.ndim == 2:
            return super().pdf(x.T)
        return super().pdf(x)


class ScipyDirichlet(ShapedDistribution):
    """
    This class allows distributions having a non-empty shape.
    """
    def __init__(self, alpha: RealArray):
        shape = alpha[..., -1].shape
        rvs_shape = (alpha.shape[-1],)
        dtype = alpha.dtype
        objects = np.empty(shape, dtype=np.object_)
        for i in np.ndindex(shape):
            objects[i] = ScipyDirichletFixRVsAndPDF(alpha[i])
        super().__init__(shape, rvs_shape, dtype, objects)

    def pdf(self, x: ComplexArray) -> RealArray:
        x = x.astype(np.float64)
        y = np.sum(x, axis=-1)
        if not np.allclose(y, np.ones(y.shape), atol=1e-5, rtol=0):
            raise ValueError
        return super().pdf(x)
