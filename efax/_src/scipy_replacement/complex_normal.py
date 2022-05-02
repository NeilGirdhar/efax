from __future__ import annotations

from typing import Optional, Type, TypeVar

import numpy as np
from numpy.random import Generator
from tjax import ComplexArray, ComplexNumeric, RealArray, RealNumeric, ShapeLike

from .multivariate_normal import ScipyMultivariateNormal, ScipyMultivariateNormalUnvectorized
from .shaped_distribution import ShapedDistribution

__all__ = ['ScipyComplexNormal']


class ScipyComplexNormalUnvectorized:
    """
    Represents an array of univariate complex normal distributions.
    """
    def __init__(self,
                 mean: ComplexNumeric,
                 variance: RealNumeric,
                 pseudo_variance: ComplexNumeric):
        super().__init__()
        self.mean: ComplexArray = np.asarray(mean)
        self.variance: RealArray = np.asarray(variance)
        self.pseudo_variance: ComplexArray = np.asarray(pseudo_variance)
        if not issubclass(self.variance.dtype.type, np.floating):
            raise TypeError(f"The variance {self.variance} has non-real "
                            f"dtype {self.variance.dtype}.")
        if np.any(np.abs(self.pseudo_variance) > self.variance):
            raise ValueError(f"The pseudo-variance {pseudo_variance} "
                             f"is bigger than the variance {variance}.")
        if not (self.mean.shape == self.variance.shape
                == self.pseudo_variance.shape):
            raise ValueError("Shape mismatch.")

    # New methods ----------------------------------------------------------------------------------
    def pdf(self, z: ComplexNumeric, out: None = None) -> RealNumeric:
        zr = np.stack([z.real, z.imag], axis=-1)
        return self.as_multivariate_normal().pdf(zr)

    def rvs(self, size: ShapeLike = (), random_state: Optional[Generator] = None) -> ComplexArray:
        if isinstance(size, int):
            size = (size,)
        if random_state is None:
            random_state = np.random.default_rng()
        xy_rvs = random_state.multivariate_normal(mean=self._multivariate_normal_mean(),
                                                  cov=self._multivariate_normal_cov(),
                                                  size=size)
        return xy_rvs[..., 0] + 1j * xy_rvs[..., 1]

    def entropy(self) -> RealArray:
        return self.as_multivariate_normal().entropy()

    def as_multivariate_normal(self) -> ScipyMultivariateNormalUnvectorized:
        mv_mean = self._multivariate_normal_mean()
        mv_cov = self._multivariate_normal_cov()
        return ScipyMultivariateNormalUnvectorized(mean=mv_mean, cov=mv_cov)

    # Private methods ------------------------------------------------------------------------------
    def _multivariate_normal_mean(self) -> RealArray:
        return np.stack([self.mean.real, self.mean.imag], axis=-1)

    def _multivariate_normal_cov(self) -> RealArray:
        cov_sum = self.variance + self.pseudo_variance
        cov_diff = self.variance - self.pseudo_variance
        xx = 0.5 * cov_sum.real
        xy = 0.5 * -cov_diff.imag
        yx = 0.5 * cov_sum.imag
        yy = 0.5 * cov_diff.real
        xx_xy = np.stack([xx, xy], axis=-1)
        yx_yy = np.stack([yx, yy], axis=-1)
        return np.stack([xx_xy, yx_yy], axis=-2)


class ScipyComplexNormal(ShapedDistribution):
    """
    This class allows distributions having a non-empty shape.
    """
    def __init__(self,
                 mean: Optional[ComplexArray] = None,
                 variance: Optional[RealArray] = None,
                 pseudo_variance: Optional[ComplexArray] = None):
        if mean is not None:
            shape = mean.shape
        elif variance is not None:
            shape = variance.shape
        elif pseudo_variance is not None:
            shape = pseudo_variance.shape
        else:
            raise ValueError
        dtype = np.result_type(*[x.dtype
                                 for x in [mean, variance, pseudo_variance]
                                 if x is not None])
        rvs_shape = ()
        if mean is None:
            mean = np.zeros(shape, dtype=dtype)
        if variance is None:
            variance = np.array(np.ones(shape, dtype=dtype).real)
        if pseudo_variance is None:
            pseudo_variance = np.zeros(shape, dtype=dtype)

        objects = np.empty(shape, dtype=np.object_)
        for i in np.ndindex(*shape):
            objects[i] = ScipyComplexNormalUnvectorized(mean[i], variance[i], pseudo_variance[i])
        super().__init__(shape, rvs_shape, dtype, objects)

    _T = TypeVar('_T', bound='ScipyComplexNormal')

    @classmethod
    def init_using_angle(cls: Type[_T],
                         mean: ComplexNumeric,
                         variance: RealNumeric,
                         angle: RealNumeric,
                         polarization: ComplexNumeric) -> _T:
        r: ComplexArray = (polarization *
                           np.exp(1j * 2 * np.pi * angle * 2))  # type: ignore[assignment]
        mean_array = np.array(mean)
        variance_array = np.array(variance)
        pseudo_variance_array = np.array(r * variance)
        return cls(mean_array, variance_array, pseudo_variance_array)

    def as_multivariate_normal(self) -> ScipyMultivariateNormal:
        objects = np.empty(self.shape, dtype=np.object_)
        for i in np.ndindex(*self.shape):
            objects[i] = self.objects[i].as_multivariate_normal()
        return ScipyMultivariateNormal(self.shape, self.rvs_shape, self.real_dtype, objects)

    @property
    def mean(self) -> ComplexArray:
        retval = np.empty(self.shape, dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            retval[i] = self.objects[i].mean
        return retval

    @property
    def variance(self) -> ComplexArray:
        retval = np.empty(self.shape, dtype=self.real_dtype)
        for i in np.ndindex(*self.shape):
            retval[i] = self.objects[i].variance
        return retval

    @property
    def pseudo_variance(self) -> ComplexArray:
        retval = np.empty(self.shape, dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            retval[i] = self.objects[i].pseudo_variance
        return retval
