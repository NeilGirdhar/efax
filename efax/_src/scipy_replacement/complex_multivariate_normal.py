from __future__ import annotations

from math import log, pi
from typing import Any, Optional, Tuple

import numpy as np
from numpy.linalg import det, inv
from numpy.random import Generator
from tjax import ComplexArray, RealArray, Shape

__all__ = ['ScipyComplexMultivariateNormal']


class ScipyComplexMultivariateNormal:

    def __init__(self, mean: ComplexArray, variance: RealArray, pseudo_variance: ComplexArray):
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
        if not np.allclose(variance, variance.T.conj()):
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
        return ((-mu.conj() @ precision @ mu).real
                - (mu @ pseudo_precision @ mu).real
                + 0.5 * log(det_s)
                - 0.5 * log(det_h)
                + self.size * log(pi)).real  # type: ignore

    def natural_parameters(self) -> Tuple[ComplexArray, RealArray, ComplexArray]:
        r, p_c = self._r_and_p_c()
        p_inv_c = inv(p_c)  # type: ignore
        precision = -p_inv_c
        pseudo_precision = r.T @ p_inv_c
        eta = -2.0 * ((precision.T @ self.mean.conj())
                      + pseudo_precision @ self.mean)
        return (eta, precision, pseudo_precision)

    def natural_to_sample(self,
                          eta: ComplexArray,
                          precision: RealArray,
                          pseudo_precision: ComplexArray) -> (
                              Tuple[ComplexArray, RealArray, ComplexArray]):
        inv_precision = inv(precision)  # type: ignore
        r = -(pseudo_precision @ inv_precision).T
        s = inv(r.conj() @ r - np.eye(self.size)) @ inv_precision  # type: ignore
        u = (r @ s).conj()
        k = inv_precision.T @ pseudo_precision
        l_eta = 0.5 * inv(k @ k.conj() - np.eye(self.size)) @ inv_precision.T @ eta  # type: ignore
        mu = l_eta.conj() - (inv_precision.T @ pseudo_precision).conj() @ l_eta
        return mu, s, u

    # https://github.com/PyCQA/pylint/issues/4326
    # pylint: disable=unsubscriptable-object
    def pdf(self, z: ComplexArray, out: None = None) -> np.floating[Any]:
        log_normalizer = self.log_normalizer()
        eta, precision, pseudo_precision = self.natural_parameters()
        # https://github.com/numpy/numpy/issues/19318
        return np.exp((eta @ z).real
                      + (z.conj() @ precision @ z).real
                      + (z @ pseudo_precision @ z).real
                      - log_normalizer)  # type: ignore

    def rvs(self, size: Shape = (), random_state: Optional[Generator] = None) -> RealArray:
        if random_state is None:
            random_state = np.random.default_rng()
        xy_rvs = random_state.multivariate_normal(mean=self._multivariate_normal_mean(),
                                                  cov=self._multivariate_normal_cov(),
                                                  size=size)
        return xy_rvs[..., :self.size] + 1j * xy_rvs[..., self.size:]

    # Private methods ------------------------------------------------------------------------------
    def _r_and_p_c(self) -> Tuple[ComplexArray, RealArray]:
        r = self.pseudo_variance.conj().T @ inv(self.variance)  # type: ignore
        p_c = self.variance - (r @ self.pseudo_variance).conj()
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
