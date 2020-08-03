from __future__ import annotations

from math import log, pi
from numbers import Real
from typing import Optional, Tuple

import numpy as np
from numpy.random import Generator
from scipy.stats import multivariate_normal
from tjax import Shape

__all__ = ['ScipyComplexNormal']


class ScipyComplexNormal:

    """
    Represents an array of univariate complex normal distributions.
    """

    def __init__(self, mean: np.ndarray, variance: np.ndarray, pseudo_variance: np.ndarray):
        self.mean = np.asarray(mean)
        self.variance = np.asarray(variance)
        self.pseudo_variance = np.asarray(pseudo_variance)
        if not issubclass(self.variance.dtype.type, Real):
            raise TypeError(f"The variance {self.variance} has non-real "
                            f"dtype {self.variance.dtype}.")
        if np.any(np.abs(self.pseudo_variance) > self.variance):
            raise ValueError(f"The pseudo-variance {pseudo_variance} "
                             f"is bigger than the variance {variance}.")
        if not (self.mean.shape == self.variance.shape
                == self.pseudo_variance.shape):
            raise ValueError("Shape mismatch.")

    # New methods ----------------------------------------------------------------------------------
    @classmethod
    def init_using_angle(cls, mean: np.ndarray, variance: np.ndarray, angle: np.ndarray,
                         polarization: np.ndarray) -> ScipyComplexNormal:
        r = polarization * np.exp(1j * 2 * np.pi * angle * 2)
        return cls(mean, variance, r * variance)

    def log_normalizer(self) -> np.ndarray:
        _, precision, pseudo_precision = self.natural_parameters()
        mu = self.mean
        det_s = self.variance.real
        det_h = -precision.real
        return (-mu.conjugate() * mu * precision
                - mu * mu * pseudo_precision
                + 0.5 * log(det_s)
                - 0.5 * log(det_h)
                + log(pi)).real

    def natural_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r, p_c = self._r_and_p_c()
        p_inv_c = 1.0 / p_c
        precision = -p_inv_c
        pseudo_precision = r * p_inv_c
        return (-2.0 * (precision * self.mean.conjugate()
                        + pseudo_precision * self.mean),
                precision,
                pseudo_precision)

    def natural_to_sample(self,
                          eta: np.ndarray,
                          precision: np.ndarray,
                          pseudo_precision: np.ndarray) -> (
                              Tuple[np.ndarray, np.ndarray, np.ndarray]):
        r = -pseudo_precision / precision
        s = 1.0 / ((r.conjugate() * r - 1.0) * precision)
        u = (r * s).conjugate()
        k = pseudo_precision / precision
        l_eta = 0.5 * eta / ((k * k.conjugate() - 1.0) * precision)
        mu = (l_eta.conjugate()
              - (pseudo_precision / precision).conjugate() * l_eta)
        return mu, s, u

    def pdf(self, z: np.ndarray, out: None = None) -> np.ndarray:
        log_normalizer = self.log_normalizer()
        eta, precision, pseudo_precision = self.natural_parameters()
        return np.exp((eta * z).real
                      + (z.conjugate() * z * precision).real
                      + (z * z * pseudo_precision).real
                      - log_normalizer)

    def rvs(self, size: Shape = (), random_state: Optional[Generator] = None) -> np.ndarray:
        if isinstance(size, int):
            size = (size,)
        if random_state is None:
            random_state = np.random.default_rng()
        # Work around limitation https://github.com/numpy/numpy/issues/15530
        mv_mean = self._multivariate_normal_mean()
        mv_cov = self._multivariate_normal_cov()
        retval = np.empty(self.mean.shape + size, dtype=self.mean.dtype)
        for i in np.ndindex(self.mean.shape):
            mvn = multivariate_normal(mean=mv_mean[i], cov=mv_cov[i])
            xy_rvs = mvn.rvs(size=size)
            retval[i] = xy_rvs[..., 0] + 1j * xy_rvs[..., 1]
        return retval

    def entropy(self) -> np.ndarray:
        # Work around limitation https://github.com/numpy/numpy/issues/15530
        mv_mean = self._multivariate_normal_mean()
        mv_cov = self._multivariate_normal_cov()
        real_dtype = np.zeros((), dtype=self.mean.dtype).real.dtype
        retval = np.zeros(self.mean.shape, dtype=real_dtype)
        for i in np.ndindex(self.mean.shape):
            mvn = multivariate_normal(mean=mv_mean[i], cov=mv_cov[i])
            retval[i] = mvn.entropy()
        return retval

    # Private methods ------------------------------------------------------------------------------
    def _r_and_p_c(self) -> Tuple[np.ndarray, np.ndarray]:
        r = self.pseudo_variance.conjugate() / self.variance
        p_c = self.variance - (r * self.pseudo_variance).conjugate()
        return r, p_c

    def _multivariate_normal_mean(self) -> np.ndarray:
        return np.stack([self.mean.real, self.mean.imag], axis=-1)

    def _multivariate_normal_cov(self) -> np.ndarray:
        cov_sum = self.variance + self.pseudo_variance
        cov_diff = self.variance - self.pseudo_variance
        xx = 0.5 * cov_sum.real
        xy = 0.5 * -cov_diff.imag
        yx = 0.5 * cov_sum.imag
        yy = 0.5 * cov_diff.real
        xx_xy = np.stack([xx, xy], axis=-1)
        yx_yy = np.stack([yx, yy], axis=-1)
        return np.stack([xx_xy, yx_yy], axis=-2)
