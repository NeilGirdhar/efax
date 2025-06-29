from __future__ import annotations

from typing import Self

import numpy as np
from numpy.random import Generator
from tjax import NumpyComplexArray, NumpyComplexNumeric, NumpyRealArray, NumpyRealNumeric, ShapeLike
from typing_extensions import override

from .multivariate_normal import ScipyMultivariateNormal, ScipyMultivariateNormalUnvectorized
from .shaped_distribution import ShapedDistribution


class ScipyComplexNormalUnvectorized:
    """Represents an array of univariate complex normal distributions."""
    @override
    def __init__(self,
                 mean: NumpyComplexNumeric,
                 variance: NumpyRealNumeric,
                 pseudo_variance: NumpyComplexNumeric) -> None:
        super().__init__()
        self.mean: NumpyComplexArray = np.asarray(mean)
        self.variance: NumpyRealArray = np.asarray(variance)
        self.pseudo_variance: NumpyComplexArray = np.asarray(pseudo_variance)
        if not np.issubdtype(self.variance.dtype, np.floating):
            msg = f"The variance {self.variance} has non-real dtype {self.variance.dtype}."
            raise TypeError(msg)
        if np.any(np.abs(self.pseudo_variance) > self.variance):
            msg = f"The pseudo-variance {pseudo_variance} is bigger than the variance {variance}."
            raise ValueError(msg)
        if not (self.mean.shape == self.variance.shape
                == self.pseudo_variance.shape):
            msg = "Shape mismatch."
            raise ValueError(msg)

    def pdf(self, z: NumpyComplexNumeric, out: None = None) -> NumpyRealNumeric:
        zr: NumpyRealArray = np.stack([np.real(z), np.imag(z)], axis=-1)  # type: ignore[list-item]
        return self.as_multivariate_normal().pdf(zr)

    def rvs(self, size: ShapeLike = (), random_state: Generator | None = None) -> NumpyComplexArray:
        if isinstance(size, int):
            size = (size,)
        if random_state is None:
            random_state = np.random.default_rng()
        xy_rvs = random_state.multivariate_normal(mean=self._multivariate_normal_mean(),
                                                  cov=self._multivariate_normal_cov(),
                                                  size=size)
        return xy_rvs[..., 0] + 1j * xy_rvs[..., 1]

    def entropy(self) -> float:
        return self.as_multivariate_normal().entropy().item()

    def as_multivariate_normal(self) -> ScipyMultivariateNormalUnvectorized:
        mv_mean = self._multivariate_normal_mean()
        mv_cov = self._multivariate_normal_cov()
        return ScipyMultivariateNormalUnvectorized(mean=mv_mean, cov=mv_cov)

    def _multivariate_normal_mean(self) -> NumpyRealArray:
        return np.stack([np.real(self.mean), np.imag(self.mean)], axis=-1)

    def _multivariate_normal_cov(self) -> NumpyRealArray:
        xx = self.variance + np.real(self.pseudo_variance)
        xy = np.imag(self.pseudo_variance)
        yy = self.variance - np.real(self.pseudo_variance)
        xx_xy = np.stack([xx, xy], axis=-1)
        yx_yy = np.stack([xy, yy], axis=-1)
        return 0.5 * np.stack([xx_xy, yx_yy], axis=-2)


class ScipyComplexNormal(
        ShapedDistribution[ScipyComplexNormalUnvectorized]):  # type: ignore  # pyright: ignore
    """This class allows distributions having a non-empty shape."""
    @override
    def __init__(self,
                 mean: NumpyComplexArray | None = None,
                 variance: NumpyRealArray | None = None,
                 pseudo_variance: NumpyComplexArray | None = None) -> None:
        if mean is not None:
            shape = mean.shape
        elif variance is not None:
            shape = variance.shape
        elif pseudo_variance is not None:
            shape = pseudo_variance.shape
        else:
            raise ValueError
        dtype = np.result_type(*[x.dtype
                                 for x in (mean, variance, pseudo_variance)
                                 if x is not None])
        rvs_shape = ()
        if mean is None:
            mean = np.zeros(shape, dtype=dtype)
        if variance is None:
            variance = np.real(np.ones(shape, dtype=dtype))
        if pseudo_variance is None:
            pseudo_variance = np.zeros(shape, dtype=dtype)

        objects = np.empty(shape, dtype=ScipyComplexNormalUnvectorized)
        for i in np.ndindex(*shape):
            objects[i] = ScipyComplexNormalUnvectorized(mean[i], variance[i], pseudo_variance[i])
        super().__init__(shape, rvs_shape, dtype, objects)

    @classmethod
    def init_using_angle(cls,
                         mean: NumpyComplexNumeric,
                         variance: NumpyRealNumeric,
                         angle: NumpyRealNumeric,
                         polarization: NumpyComplexNumeric) -> Self:
        polarization_np: NumpyComplexArray = np.asarray(polarization)
        r: NumpyComplexArray = polarization_np * np.exp(1j * 2 * np.pi * angle * 2)
        mean_array = np.asarray(mean)
        variance_array = np.asarray(variance)
        pseudo_variance_array = r * np.asarray(variance)
        return cls(mean_array, variance_array, pseudo_variance_array)

    def as_multivariate_normal(self) -> ScipyMultivariateNormal:
        objects = np.empty(self.shape, dtype=ScipyMultivariateNormalUnvectorized)
        for i in np.ndindex(*self.shape):
            this_object = self.objects[i]
            assert isinstance(this_object, ScipyComplexNormalUnvectorized)
            objects[i] = this_object.as_multivariate_normal()
        return ScipyMultivariateNormal(self.shape, self.rvs_shape, self.real_dtype, objects)

    @property
    def mean(self) -> NumpyComplexArray:
        retval = np.empty(self.shape, dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            retval[i] = self.objects[i].mean
        return retval

    @property
    def variance(self) -> NumpyComplexArray:
        retval = np.empty(self.shape, dtype=self.real_dtype)
        for i in np.ndindex(*self.shape):
            this_object = self.objects[i]
            assert isinstance(this_object, ScipyComplexNormalUnvectorized)
            retval[i] = this_object.variance
        return retval

    @property
    def pseudo_variance(self) -> NumpyComplexArray:
        retval = np.empty(self.shape, dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            this_object = self.objects[i]
            assert isinstance(this_object, ScipyComplexNormalUnvectorized)
            retval[i] = this_object.pseudo_variance
        return retval
