from math import log, pi
from typing import Optional, Tuple

import numpy as np
from numpy.linalg import det, inv
from numpy.random import Generator
from tjax import Shape

__all__ = ['ScipyComplexMultivariateNormal']


class ScipyComplexMultivariateNormal:

    def __init__(self, mean: np.ndarray, variance: np.ndarray, pseudo_variance: np.ndarray):
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
        if not np.allclose(variance, variance.T.conj()):
            raise ValueError("The variance is not Hermitian.")
        if not np.allclose(pseudo_variance, pseudo_variance.T):
            raise ValueError("The pseudo-variance is not symmetric.")

    # New methods ----------------------------------------------------------------------------------
    def log_normalizer(self) -> np.ndarray:
        mu = self.mean
        _, precision, pseudo_precision = self.natural_parameters()
        det_s = det(self.variance).real
        det_h = det(-precision).real
        return (-mu.conj().dot(precision.dot(mu)).real
                - mu.dot(pseudo_precision.dot(mu)).real
                + 0.5 * log(det_s)
                - 0.5 * log(det_h)
                + self.size * log(pi)).real

    def natural_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r, p_c = self._r_and_p_c()
        p_inv_c = inv(p_c)
        precision = -p_inv_c
        pseudo_precision = r.T @ p_inv_c
        eta = -2.0 * (precision.T.dot(self.mean.conj())
                      + pseudo_precision.dot(self.mean))
        return (eta, precision, pseudo_precision)

    def natural_to_sample(self,
                          eta: np.ndarray,
                          precision: np.ndarray,
                          pseudo_precision: np.ndarray) -> (
                              Tuple[np.ndarray, np.ndarray, np.ndarray]):
        inv_precision = inv(precision)
        r = -(pseudo_precision @ inv_precision).T
        s = inv(r.conj() @ r - np.eye(self.size)) @ inv_precision
        u = (r @ s).conj()
        k = inv_precision.T @ pseudo_precision
        l_eta = 0.5 * inv(k @ k.conj()
                          - np.eye(self.size)).dot(inv_precision.T.dot(eta))
        mu = (l_eta.conj()
              - (inv_precision.T @ pseudo_precision).conj().dot(l_eta))
        return mu, s, u

    def pdf(self, z: np.ndarray, out: None = None) -> np.ndarray:
        log_normalizer = self.log_normalizer()
        eta, precision, pseudo_precision = self.natural_parameters()
        return np.exp((eta.dot(z)
                       + z.conj().dot(precision).dot(z)
                       + z.dot(pseudo_precision).dot(z)).real
                      - log_normalizer)

    def rvs(self, size: Shape = (), random_state: Optional[Generator] = None) -> np.ndarray:
        if random_state is None:
            random_state = np.random.default_rng()
        xy_rvs = random_state.multivariate_normal(mean=self._multivariate_normal_mean(),
                                                  cov=self._multivariate_normal_cov(),
                                                  size=size)
        return xy_rvs[..., :self.size] + 1j * xy_rvs[..., self.size:]

    # Private methods ------------------------------------------------------------------------------
    def _r_and_p_c(self) -> Tuple[np.ndarray, np.ndarray]:
        r = self.pseudo_variance.conj().T @ inv(self.variance)
        p_c = self.variance - (r @ self.pseudo_variance).conj()
        return r, p_c

    def _multivariate_normal_mean(self) -> np.ndarray:
        return np.concatenate([self.mean.real, self.mean.imag])

    def _multivariate_normal_cov(self) -> np.ndarray:
        cov_sum = self.variance + self.pseudo_variance
        cov_diff = self.variance - self.pseudo_variance
        xx = 0.5 * cov_sum.real
        xy = 0.5 * -cov_diff.imag
        yx = 0.5 * cov_sum.imag
        yy = 0.5 * cov_diff.real
        return np.block([[xx, xy], [yx, yy]])
