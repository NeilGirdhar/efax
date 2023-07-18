from __future__ import annotations

import numpy as np
import scipy.special
import scipy.stats as ss
from numpy.random import Generator
from tjax import NumpyComplexArray, NumpyRealArray, ShapeLike
from typing_extensions import override

from .shaped_distribution import ShapedDistribution

__all__ = ['ScipyDirichlet', 'ScipyGeneralizedDirichlet']


scipy_beta = scipy.special.beta


# pylint: disable=protected-access
mvd: type = ss._multivariate.dirichlet_frozen  # noqa: SLF001
class ScipyDirichletFixRVsAndPDF(mvd):
    """This class repairs dirichlet.

    See https://github.com/scipy/scipy/issues/6005 and https://github.com/scipy/scipy/issues/6006.
    """
    @override
    def rvs(self,  # type: ignore[misc]
            size: ShapeLike | None = None,
            random_state: Generator | None = None) -> NumpyRealArray:
        if size is None:
            size = ()
        # This somehow fixes the behaviour of rvs.
        return super().rvs(size=size, random_state=random_state)  # pyright: ignore

    @override
    def pdf(self, x: NumpyRealArray) -> NumpyRealArray:  # type: ignore[misc]
        if x.ndim == 2:  # noqa: PLR2004
            return super().pdf(x.T)
        return super().pdf(x)


class ScipyDirichlet(ShapedDistribution[ScipyDirichletFixRVsAndPDF]):
    """This class allows distributions having a non-empty shape."""
    @override
    def __init__(self, alpha: NumpyRealArray):
        shape = alpha[..., -1].shape
        rvs_shape = (alpha.shape[-1],)
        dtype = alpha.dtype
        objects = np.empty(shape, dtype=ScipyDirichletFixRVsAndPDF)
        for i in np.ndindex(shape):
            objects[i] = ScipyDirichletFixRVsAndPDF(alpha[i])
        super().__init__(shape, rvs_shape, dtype, objects)

    @override
    def pdf(self, x: NumpyComplexArray) -> NumpyRealArray:
        x = x.astype(np.float64)
        y = np.sum(x, axis=-1)
        if not np.all(np.isclose(y, np.ones(y.shape), atol=1e-5, rtol=0)):
            raise ValueError
        return super().pdf(x)


class ScipyGeneralizedDirichlet:
    @override
    def __init__(self, alpha: NumpyRealArray, beta: NumpyRealArray):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def pdf(self, x: NumpyRealArray) -> NumpyRealArray:
        alpha_roll = np.roll(self.alpha, -1, axis=-1)
        alpha_roll[..., -1] = 0.0
        gamma = -np.diff(self.beta, append=1.0) - alpha_roll
        cs_x = np.cumsum(x, axis=-1)
        terms = x ** (self.alpha - 1.0) * (1.0 - cs_x) ** gamma / scipy_beta(self.alpha, self.beta)
        return np.prod(terms, axis=-1)

    def rvs(self, size: ShapeLike = (), random_state: Generator | None = None) -> NumpyComplexArray:
        sample_size: tuple[int, ...] = (((size,) if isinstance(size, int) else tuple(size))
                                        + self.alpha.shape)
        if random_state is None:
            random_state = np.random.default_rng()
        dimensions = self.alpha.shape[-1]
        beta_samples = random_state.beta(self.alpha, self.beta, size=sample_size)
        q = np.zeros(beta_samples.shape[:-1])
        for i in range(dimensions):
            beta_samples[..., i] *= 1 - q
            q += beta_samples[..., i]
        return beta_samples

    def entropy(self) -> NumpyRealArray:
        raise RuntimeError
