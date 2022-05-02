from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.random import Generator
from tjax import ComplexArray, RealArray, ShapeLike

from .multivariate_normal import ScipyMultivariateNormal, ScipyMultivariateNormalUnvectorized
from .shaped_distribution import ShapedDistribution

__all__ = ['ScipyComplexMultivariateNormal']


class ScipyComplexMultivariateNormalUnvectorized:
    """
    Represents a multivariate complex normal distribution.
    """
    def __init__(self, mean: ComplexArray, variance: ComplexArray, pseudo_variance: ComplexArray):
        super().__init__()
        self.size = mean.shape[0]
        self.mean = mean
        self.variance = variance
        self.pseudo_variance = pseudo_variance
        if mean.shape != (self.size,):
            raise ValueError(f"Mean has shape {mean.shape} instead of {(self.size,)}.")
        if variance.shape != (self.size, self.size):
            raise ValueError("The variance has shape "
                             f"{variance.shape} "
                             f"instead of {(self.size, self.size)}.")
        if pseudo_variance.shape != (self.size, self.size):
            raise ValueError("The pseudo-variance has shape "
                             f"{pseudo_variance.shape} "
                             f"instead of {(self.size, self.size)}.")
        if not np.all(np.linalg.eigvals(variance) >= 0):
            raise ValueError("The variance is not positive semidefinite.")
        if not np.allclose(variance, variance.T.conjugate()):
            raise ValueError("The variance is not Hermitian.")
        if not np.allclose(pseudo_variance, pseudo_variance.T):
            raise ValueError("The pseudo-variance is not symmetric.")

    # New methods ----------------------------------------------------------------------------------
    def pdf(self, z: ComplexArray, out: None = None) -> np.floating[Any]:
        zr = np.concatenate([z.real, z.imag], axis=-1)
        return self.as_multivariate_normal().pdf(zr)

    def rvs(self, size: ShapeLike = (), random_state: Optional[Generator] = None) -> ComplexArray:
        if isinstance(size, int):
            size = (size,)
        if random_state is None:
            random_state = np.random.default_rng()
        xy_rvs = random_state.multivariate_normal(mean=self._multivariate_normal_mean(),
                                                  cov=self._multivariate_normal_cov(),
                                                  size=size)
        return xy_rvs[..., :self.size] + 1j * xy_rvs[..., self.size:]

    def entropy(self) -> RealArray:
        return self.as_multivariate_normal().entropy()

    def as_multivariate_normal(self) -> ScipyMultivariateNormalUnvectorized:
        mv_mean = self._multivariate_normal_mean()
        mv_cov = self._multivariate_normal_cov()
        return ScipyMultivariateNormalUnvectorized(mean=mv_mean, cov=mv_cov)

    # Private methods ------------------------------------------------------------------------------
    def _multivariate_normal_mean(self) -> RealArray:
        "Return the mean of a corresponding real distribution with double the size."
        return np.concatenate([self.mean.real, self.mean.imag])

    def _multivariate_normal_cov(self) -> RealArray:
        "Return the covariance of a corresponding real distribution with double the size."
        cov_sum = self.variance + self.pseudo_variance
        cov_diff = self.variance - self.pseudo_variance
        xx = 0.5 * cov_sum.real
        xy = 0.5 * -cov_diff.imag
        yx = 0.5 * cov_sum.imag
        yy = 0.5 * cov_diff.real
        return np.block([[xx, xy], [yx, yy]])


class ScipyComplexMultivariateNormal(ShapedDistribution):
    """
    This class allows distributions having a non-empty shape.
    """
    def __init__(self,
                 mean: Optional[ComplexArray] = None,
                 variance: Optional[ComplexArray] = None,
                 pseudo_variance: Optional[ComplexArray] = None):
        if mean is not None:
            shape = mean.shape[:-1]
            dimensions = mean.shape[-1]
        elif variance is not None:
            shape = variance.shape[:-2]
            dimensions = variance.shape[-1]
        elif pseudo_variance is not None:
            shape = pseudo_variance.shape[:-2]
            dimensions = pseudo_variance.shape[-1]
        else:
            raise ValueError
        dtype = np.result_type(*[x.dtype
                                 for x in [mean, variance, pseudo_variance]
                                 if x is not None])
        rvs_shape = (dimensions,)
        if mean is None:
            mean = np.zeros(shape + (dimensions,), dtype=dtype)
        if variance is None:
            variance = np.array(np.tile(np.eye(dimensions), shape + (1, 1)))
        if pseudo_variance is None:
            pseudo_variance = np.zeros(shape + (dimensions, dimensions), dtype=dtype)

        objects = np.empty(shape, dtype=np.object_)
        for i in np.ndindex(*shape):
            objects[i] = ScipyComplexMultivariateNormalUnvectorized(mean[i], variance[i],
                                                                    pseudo_variance[i])
        super().__init__(shape, rvs_shape, dtype, objects)

    def as_multivariate_normal(self) -> ScipyMultivariateNormal:
        objects = np.empty(self.shape, dtype=np.object_)
        for i in np.ndindex(*self.shape):
            objects[i] = self.objects[i].as_multivariate_normal()
        return ScipyMultivariateNormal(self.shape, self.rvs_shape, self.real_dtype, objects)

    @property
    def mean(self) -> ComplexArray:
        retval = np.empty(self.shape + self.rvs_shape, dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            retval[i] = self.objects[i].mean
        return retval

    @property
    def variance(self) -> ComplexArray:
        retval = np.empty(self.shape + self.rvs_shape + self.rvs_shape, dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            retval[i] = self.objects[i].variance
        return retval

    @property
    def pseudo_variance(self) -> ComplexArray:
        retval = np.empty(self.shape + self.rvs_shape + self.rvs_shape, dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            retval[i] = self.objects[i].pseudo_variance
        return retval
