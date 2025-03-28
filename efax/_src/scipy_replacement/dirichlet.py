from __future__ import annotations

from typing import Any

import numpy as np
import optype.numpy as onp
import scipy.stats as ss
from numpy.random import Generator
from scipy.special import beta as scipy_beta
from tjax import NumpyComplexArray, NumpyRealArray, ShapeLike
from typing_extensions import override

from .shaped_distribution import ShapedDistribution


class ScipyDirichletFixRVsAndPDF:
    """This class repairs dirichlet.

    See https://github.com/scipy/scipy/issues/6005 and https://github.com/scipy/scipy/issues/6006.
    """
    def __init__(self, alpha: NumpyRealArray) -> None:
        super().__init__()
        self.distribution = ss.dirichlet(alpha=alpha)

    def pdf(self, x: onp.ToFloatND) -> np.float64:
        x = np.asarray(x)
        if x.ndim == 2:  # noqa: PLR2004
            return np.float64(self.distribution.pdf(x.T))
        return np.float64(self.distribution.pdf(x))

    def rvs(self,
            size: Any = None,
            random_state: Any = None
            ) -> onp.ArrayND[np.float64]:
        if size is None:
            size = ()
        # This somehow fixes the behaviour of rvs.
        return self.distribution.rvs(size=size, random_state=random_state)

    def entropy(self) -> NumpyRealArray:
        return np.asarray(self.distribution.entropy())


class ScipyDirichlet(ShapedDistribution[ScipyDirichletFixRVsAndPDF]):
    """This class allows distributions having a non-empty shape."""
    @override
    def __init__(self, alpha: NumpyRealArray) -> None:
        shape = alpha[..., -1].shape
        rvs_shape = (alpha.shape[-1],)
        dtype = alpha.dtype
        objects = np.empty(shape, dtype=ScipyDirichletFixRVsAndPDF)
        for i in np.ndindex(shape):
            objects[i] = ScipyDirichletFixRVsAndPDF(alpha[i])
        super().__init__(shape, rvs_shape, dtype, objects)

    @override
    def pdf(self, x: NumpyComplexArray) -> NumpyRealArray:
        x = np.astype(x, np.float64)
        y = np.sum(x, axis=-1)
        if not np.all(np.isclose(y, np.ones(y.shape), atol=1e-5, rtol=0)):
            raise ValueError
        return super().pdf(x)


class ScipyGeneralizedDirichlet:
    @override
    def __init__(self, alpha: NumpyRealArray, beta: NumpyRealArray) -> None:
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
