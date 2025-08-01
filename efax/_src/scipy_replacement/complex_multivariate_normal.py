from __future__ import annotations

import numpy as np
from numpy.random import Generator
from tjax import NumpyComplexArray, NumpyRealArray, ShapeLike
from typing_extensions import override

from .multivariate_normal import ScipyMultivariateNormal, ScipyMultivariateNormalUnvectorized
from .shaped_distribution import ShapedDistribution


class ScipyComplexMultivariateNormalUnvectorized:
    """Represents a multivariate complex normal distribution."""
    @override
    def __init__(self,
                 mean: NumpyComplexArray,
                 variance: NumpyComplexArray,
                 pseudo_variance: NumpyComplexArray) -> None:
        super().__init__()
        self.size = mean.shape[0]
        self.mean = mean
        self.variance = variance
        self.pseudo_variance = pseudo_variance
        if mean.shape != (self.size,):
            msg = f"Mean has shape {mean.shape} instead of {(self.size,)}."
            raise ValueError(msg)
        if variance.shape != (self.size, self.size):
            msg = "The variance has shape {variance.shape} instead of {(self.size, self.size)}."
            raise ValueError(msg)
        if pseudo_variance.shape != (self.size, self.size):
            msg = ("The pseudo-variance has shape {pseudo_variance.shape} "
                   f"instead of {(self.size, self.size)}.")
            raise ValueError(msg)
        if not np.all(np.linalg.eigvals(variance) >= 0):
            msg = "The variance is not positive semidefinite."
            raise ValueError(msg)
        if not np.all(np.isclose(variance, np.conj(variance.T))):
            msg = "The variance is not Hermitian."
            raise ValueError(msg)
        if not np.all(np.isclose(pseudo_variance, pseudo_variance.T)):
            msg = "The pseudo-variance is not symmetric."
            raise ValueError(msg)

    def pdf(self, z: NumpyComplexArray, out: None = None) -> float:
        zr = np.concat([np.real(z), np.imag(z)], axis=-1)
        return self.as_multivariate_normal().pdf(zr).item()

    def rvs(self, size: ShapeLike = (), random_state: Generator | None = None) -> NumpyComplexArray:
        if isinstance(size, int):
            size = (size,)
        if random_state is None:
            random_state = np.random.default_rng()
        xy_rvs = random_state.multivariate_normal(mean=self._multivariate_normal_mean(),
                                                  cov=self._multivariate_normal_cov(),
                                                  size=size)
        return xy_rvs[..., :self.size] + 1j * xy_rvs[..., self.size:]

    def entropy(self) -> float:
        return self.as_multivariate_normal().entropy().item()

    def as_multivariate_normal(self) -> ScipyMultivariateNormalUnvectorized:
        mv_mean = self._multivariate_normal_mean()
        mv_cov = self._multivariate_normal_cov()
        return ScipyMultivariateNormalUnvectorized(mean=mv_mean, cov=mv_cov)

    def _multivariate_normal_mean(self) -> NumpyRealArray:
        """Return the mean of a corresponding real distribution with double the size."""
        return np.concat((np.real(self.mean), np.imag(self.mean)))

    def _multivariate_normal_cov(self) -> NumpyRealArray:
        """Return the covariance of a corresponding real distribution with double the size."""
        cov_sum = self.variance + self.pseudo_variance
        cov_diff = self.variance - self.pseudo_variance
        xx = 0.5 * np.real(cov_sum)
        xy = 0.5 * np.imag(-cov_diff)
        yx = 0.5 * np.imag(cov_sum)
        yy = 0.5 * np.real(cov_diff)
        return np.block([[xx, xy], [yx, yy]])


class ScipyComplexMultivariateNormal(
        ShapedDistribution[
            ScipyComplexMultivariateNormalUnvectorized]):  # type: ignore # pyright: ignore
    """This class allows distributions having a non-empty shape."""
    @override
    def __init__(self,
                 mean: NumpyComplexArray | None = None,
                 variance: NumpyComplexArray | None = None,
                 pseudo_variance: NumpyComplexArray | None = None) -> None:
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
                                 for x in (mean, variance, pseudo_variance)
                                 if x is not None])
        rvs_shape = (dimensions,)
        if mean is None:
            mean = np.zeros((*shape, dimensions), dtype=dtype)
        if variance is None:
            variance = np.asarray(np.tile(np.eye(dimensions), (*shape, 1, 1)))
        if pseudo_variance is None:
            pseudo_variance = np.zeros((*shape, dimensions, dimensions), dtype=dtype)

        objects = np.empty(shape, dtype=ScipyComplexMultivariateNormalUnvectorized)
        for i in np.ndindex(*shape):
            objects[i] = ScipyComplexMultivariateNormalUnvectorized(
                    mean[i], variance[i], pseudo_variance[i])
        super().__init__(shape, rvs_shape, dtype, objects)

    def as_multivariate_normal(self) -> ScipyMultivariateNormal:
        objects = np.empty(self.shape, dtype=ScipyMultivariateNormalUnvectorized)
        for i in np.ndindex(*self.shape):
            this_object = self.objects[i]
            assert isinstance(this_object, ScipyComplexMultivariateNormalUnvectorized)
            objects[i] = this_object.as_multivariate_normal()
        return ScipyMultivariateNormal(self.shape, self.rvs_shape, self.real_dtype, objects)

    @property
    def mean(self) -> NumpyComplexArray:
        retval = np.empty(self.shape + self.rvs_shape, dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            retval[i] = self.objects[i].mean
        return retval

    @property
    def variance(self) -> NumpyComplexArray:
        retval = np.empty(self.shape + self.rvs_shape + self.rvs_shape, dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            this_object = self.objects[i]
            assert isinstance(this_object, ScipyComplexMultivariateNormalUnvectorized)
            retval[i] = this_object.variance
        return retval

    @property
    def pseudo_variance(self) -> NumpyComplexArray:
        retval = np.empty(self.shape + self.rvs_shape + self.rvs_shape, dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            this_object = self.objects[i]
            assert isinstance(this_object, ScipyComplexMultivariateNormalUnvectorized)
            retval[i] = this_object.pseudo_variance
        return retval
