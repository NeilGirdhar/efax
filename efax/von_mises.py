from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import scipy.optimize
from chex import Array
from jax import numpy as jnp
from jax.nn import softplus
from scipy.special import iv
from tjax import RealArray, Shape, dataclass

from .exponential_family import ExpectationParametrization, NaturalParametrization

__all__ = ['VonMisesFisherNP', 'VonMisesFisherEP', 'VonMisesNP', 'VonMisesEP']


@dataclass
class VonMisesFisherNP(NaturalParametrization['VonMisesFisherEP']):
    mean_times_concentration: RealArray

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean_times_concentration.shape

    def log_normalizer(self) -> RealArray:
        half_k = self.mean_times_concentration.shape[-1] * 0.5
        kappa = jnp.linalg.norm(self.mean_times_concentration, 2, axis=-1)
        return -jnp.log(kappa ** (half_k - 1.0)
                        / ((2.0 * math.pi) ** half_k
                           * iv(half_k - 1.0, kappa)))

    def to_exp(self) -> VonMisesFisherEP:
        q = self.mean_times_concentration
        kappa = jnp.linalg.norm(q, 2, axis=-1, keepdims=True)
        return VonMisesFisherEP(
            jnp.where(kappa == 0.0,
                      q,
                      q * (_a_k(q.shape[-1], kappa) / kappa)))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape)

    def sufficient_statistics(self, x: RealArray) -> VonMisesFisherEP:
        return VonMisesFisherEP(x)


@dataclass
class VonMisesFisherEP(ExpectationParametrization[VonMisesFisherNP]):
    mean: RealArray

    # Implemented methods --------------------------------------------------------------------------
    def to_nat(self) -> VonMisesFisherNP:
        p = self.mean
        k = p.shape[-1]
        mu = jnp.linalg.norm(p, 2, axis=-1)
        q = np.empty_like(p)
        for i in np.ndindex(p.shape[: -1]):
            this_p = p[i]
            this_mu = mu[i]
            kappa = VonMisesFisherEP._find_kappa(k, this_mu)
            q[i] = this_p * (kappa / this_mu)
        return VonMisesFisherNP(q)

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape())

    # Private methods ------------------------------------------------------------------------------
    @staticmethod
    def _find_kappa(k: RealArray, mu: RealArray) -> RealArray:
        assert 0 <= mu <= 1.0
        if mu == 0.0:
            return 0.0
        initial_solution = (mu * k - mu ** 3) / (1.0 - mu ** 2)

        def f(isp_kappa: RealArray) -> RealArray:
            return _a_k(k, softplus(isp_kappa)) - mu
        solution = scipy.optimize.root(f,
                                       inverse_softplus(initial_solution),
                                       tol=1e-5)
        if not solution.success:
            raise ValueError(f"Failed to find kappa because {solution.message}.")
        return softplus(solution.x)


def inverse_softplus(y: Array) -> Array:
    return jnp.where(y > 80.0,
                     y,
                     jnp.log(jnp.expm1(y)))


@dataclass
class VonMisesNP(VonMisesFisherNP):

    # Overridden methods ---------------------------------------------------------------------------
    def to_exp(self) -> VonMisesEP:
        return VonMisesEP(super().to_exp().mean)

    def sufficient_statistics(self, x: RealArray) -> VonMisesFisherEP:
        return VonMisesEP(super().sufficient_statistics(x).mean)

    # New methods ----------------------------------------------------------------------------------
    def nat_to_kappa_angle(self) -> Tuple[RealArray, RealArray]:
        kappa = np.linalg.norm(self.mean_times_concentration, axis=-1)
        angle = np.where(kappa == 0.0,
                         0.0,
                         np.arctan2(self.mean_times_concentration[..., 1],
                                    self.mean_times_concentration[..., 0]))
        return kappa, angle


@dataclass
class VonMisesEP(VonMisesFisherEP):

    # Overridden methods ---------------------------------------------------------------------------
    def to_nat(self) -> VonMisesNP:
        return VonMisesNP(super().to_nat().mean_times_concentration)


# Private functions --------------------------------------------------------------------------------
def _a_k(k: RealArray, kappa: RealArray) -> RealArray:
    half_k = k * 0.5
    return iv(half_k, kappa) / iv(half_k - 1.0, kappa)
