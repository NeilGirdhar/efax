from __future__ import annotations

import numpy as np
import scipy.special
import scipy.stats as ss
from numpy.random import Generator
from tjax import NumpyComplexArray, NumpyRealArray, ShapeLike

from .shaped_distribution import ShapedDistribution

__all__ = ['ScipyDirichlet', 'ScipyGeneralizedDirichlet']


# pylint: disable=protected-access
class ScipyDirichletFixRVsAndPDF(ss._multivariate.dirichlet_frozen):
    """
    This class repairs dirichlet.  See https://github.com/scipy/scipy/issues/6005 and
    https://github.com/scipy/scipy/issues/6006.
    """
    def rvs(self,
            size: ShapeLike | None = None,
            random_state: Generator | None = None) -> NumpyRealArray:
        if size is None:
            size = ()
        # This somehow fixes the behaviour of rvs.
        return super().rvs(size=size, random_state=random_state)

    def pdf(self, x: NumpyRealArray) -> NumpyRealArray:
        if x.ndim == 2:
            return super().pdf(x.T)
        return super().pdf(x)


class ScipyDirichlet(ShapedDistribution):
    """
    This class allows distributions having a non-empty shape.
    """
    def __init__(self, alpha: NumpyRealArray):
        shape = alpha[..., -1].shape
        rvs_shape = (alpha.shape[-1],)
        dtype = alpha.dtype
        objects = np.empty(shape, dtype=np.object_)
        for i in np.ndindex(shape):
            objects[i] = ScipyDirichletFixRVsAndPDF(alpha[i])
        super().__init__(shape, rvs_shape, dtype, objects)

    def pdf(self, x: NumpyComplexArray) -> NumpyRealArray:
        x = x.astype(np.float64)
        y = np.sum(x, axis=-1)
        if not np.allclose(y, np.ones(y.shape), atol=1e-5, rtol=0):
            raise ValueError
        return super().pdf(x)


class ScipyGeneralizedDirichlet:
    def __init__(self, alpha: NumpyRealArray, beta: NumpyRealArray):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def pdf(self, x: NumpyRealArray) -> NumpyRealArray:
        alpha_roll = np.roll(self.alpha, -1, axis=-1)
        alpha_roll[..., -1] = 0.0
        gamma = -np.diff(self.beta, append=1.0) - alpha_roll
        cs_x = np.cumsum(x, axis=-1)
        terms = x ** (self.alpha - 1.0) * (1.0 - cs_x) ** gamma / scipy.special.beta(self.alpha,
                                                                                     self.beta)
        return np.prod(terms, axis=-1)

    def rvs(self, size: ShapeLike = (), random_state: Generator | None = None) -> NumpyComplexArray:
        if isinstance(size, int):
            size = (size,)
        if random_state is None:
            random_state = np.random.default_rng()
        dimensions = self.alpha.shape[-1]
        beta_samples = random_state.beta(self.alpha, self.beta,
                                         size=(tuple(size) + self.alpha.shape))
        q = np.zeros(beta_samples.shape[:-1])
        for i in range(dimensions):
            beta_samples[..., i] *= 1 - q
            q += beta_samples[..., i]
        return beta_samples

    def entropy(self) -> NumpyRealArray:
        raise NotImplementedError
