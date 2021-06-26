from __future__ import annotations

from math import log, pi
from typing import Any, Optional, Tuple

import numpy as np
from numpy.linalg import det, inv
from numpy.random import Generator
from scipy.stats import multivariate_normal
from tjax import ComplexArray, RealArray, Shape

__all__ = ['ScipyComplexMultivariateNormal']


class ScipyComplexMultivariateNormal:
    """
    Represents a multivariate complex normal distribution.
    """
    def __init__(self, mean: ComplexArray, variance: ComplexArray, pseudo_variance: ComplexArray):
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
        if not np.all(np.linalg.eigvals(variance) >= 0):  # type: ignore
            raise ValueError("The variance is not positive semidefinite.")
        if not np.allclose(variance, variance.T.conjugate()):
            raise ValueError("The variance is not Hermitian.")
        if not np.allclose(pseudo_variance, pseudo_variance.T):
            raise ValueError("The pseudo-variance is not symmetric.")

    # New methods ----------------------------------------------------------------------------------
    def log_normalizer(self) -> RealArray:
        mu = self.mean
        _, precision, pseudo_precision = self.natural_parameters()
        det_s: RealArray = det(self.variance).real  # type: ignore
        det_h: RealArray = det(-precision).real  # type: ignore
        # https://github.com/numpy/numpy/issues/19318
        return ((-mu.conjugate() @ precision @ mu).real
                - (mu @ pseudo_precision @ mu).real
                + 0.5 * log(det_s)
                - 0.5 * log(det_h)
                + self.size * log(pi)).real  # type: ignore

    def natural_parameters(self) -> Tuple[ComplexArray, ComplexArray, ComplexArray]:
        r, p_c = self._r_and_p_c()
        p_inv_c: ComplexArray = inv(p_c)  # type: ignore
        precision = -p_inv_c
        pseudo_precision = r.T @ p_inv_c
        eta = -2.0 * ((precision.T @ self.mean.conjugate())
                      + pseudo_precision @ self.mean)
        return (eta, precision, pseudo_precision)

    def natural_to_sample(self,
                          eta: ComplexArray,
                          precision: ComplexArray,
                          pseudo_precision: ComplexArray) -> (
                              Tuple[ComplexArray, ComplexArray, ComplexArray]):
        inv_precision = inv(precision)  # type: ignore
        r = -(pseudo_precision @ inv_precision).T
        s = inv(r.conjugate() @ r - np.eye(self.size)) @ inv_precision  # type: ignore
        u = (r @ s).conjugate()
        k = inv_precision.T @ pseudo_precision
        l_eta = 0.5 * inv(k @ k.conjugate()  # type: ignore
                          - np.eye(self.size)) @ inv_precision.T @ eta
        mu = l_eta.conjugate() - (inv_precision.T @ pseudo_precision).conjugate() @ l_eta
        return mu, s, u

    # https://github.com/PyCQA/pylint/issues/4326
    # pylint: disable=unsubscriptable-object
    def pdf(self, z: ComplexArray, out: None = None) -> np.floating[Any]:
        log_normalizer = self.log_normalizer()
        eta, precision, pseudo_precision = self.natural_parameters()
        # https://github.com/numpy/numpy/issues/19318
        return np.exp((eta @ z).real
                      + (z.conjugate() @ precision @ z).real
                      + (z @ pseudo_precision @ z).real
                      - log_normalizer)  # type: ignore

    def rvs(self, size: Shape = (), random_state: Optional[Generator] = None) -> ComplexArray:
        if random_state is None:
            random_state = np.random.default_rng()
        xy_rvs = random_state.multivariate_normal(mean=self._multivariate_normal_mean(),
                                                  cov=self._multivariate_normal_cov(),
                                                  size=size)
        return xy_rvs[..., :self.size] + 1j * xy_rvs[..., self.size:]

    def entropy(self) -> RealArray:
        mv_mean = self._multivariate_normal_mean()
        mv_cov = self._multivariate_normal_cov()
        mvn = multivariate_normal(mean=mv_mean, cov=mv_cov)
        return mvn.entropy()

    # Private methods ------------------------------------------------------------------------------
    def _r_and_p_c(self) -> Tuple[ComplexArray, ComplexArray]:
        r = self.pseudo_variance.conjugate().T @ inv(self.variance)  # type: ignore
        p_c = self.variance - (r @ self.pseudo_variance).conjugate()
        return r, p_c

    def _multivariate_normal_mean(self) -> RealArray:
        return np.concatenate([self.mean.real, self.mean.imag])  # type: ignore

    def _multivariate_normal_cov(self) -> RealArray:
        cov_sum = self.variance + self.pseudo_variance
        cov_diff = self.variance - self.pseudo_variance
        xx = 0.5 * cov_sum.real
        xy = 0.5 * -cov_diff.imag
        yx = 0.5 * cov_sum.imag
        yy = 0.5 * cov_diff.real
        return np.block([[xx, xy], [yx, yy]])
