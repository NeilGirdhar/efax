from __future__ import annotations

import math
from typing import Tuple

import jax.numpy as jnp
from jax.nn import softplus
from tjax import RealArray, Shape, dataclass

from .exp_to_nat import ExpToNat
from .natural_parametrization import NaturalParametrization
from .parameter import VectorSupport, distribution_parameter
from .tools import inverse_softplus, ive

__all__ = ['VonMisesFisherNP', 'VonMisesFisherEP']


@dataclass
class VonMisesFisherNP(NaturalParametrization['VonMisesFisherEP']):
    mean_times_concentration: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean_times_concentration.shape[:-1]

    def log_normalizer(self) -> RealArray:
        half_k = self.mean_times_concentration.shape[-1] * 0.5
        kappa = jnp.linalg.norm(self.mean_times_concentration, 2, axis=-1)
        return kappa - jnp.log(kappa ** (half_k - 1.0)
                               / ((2.0 * math.pi) ** half_k * ive(half_k - 1.0, kappa)))

    def to_exp(self) -> VonMisesFisherEP:
        q = self.mean_times_concentration
        kappa = jnp.linalg.norm(q, 2, axis=-1, keepdims=True)
        return VonMisesFisherEP(
            jnp.where(kappa == 0.0,
                      q,
                      q * (_a_k(q.shape[-1], kappa) / kappa)))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(self.shape())

    def sufficient_statistics(self, x: RealArray) -> VonMisesFisherEP:
        return VonMisesFisherEP(x)

    # New methods ----------------------------------------------------------------------------------
    def to_kappa_angle(self) -> Tuple[RealArray, RealArray]:
        if self.mean_times_concentration.shape[-1] != 2:
            raise ValueError
        kappa = jnp.linalg.norm(self.mean_times_concentration, axis=-1)
        angle = jnp.where(kappa == 0.0,
                          0.0,
                          jnp.arctan2(self.mean_times_concentration[..., 1],
                                      self.mean_times_concentration[..., 0]))
        return kappa, angle


@dataclass
class VonMisesFisherEP(ExpToNat[VonMisesFisherNP, RealArray]):
    mean: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape())

    def initial_search_parameters(self) -> RealArray:
        k = self.mean.shape[-1]
        mu = jnp.linalg.norm(self.mean, 2, axis=-1)
        # 0 <= mu <= 1.0
        initial_kappa = jnp.where(mu == 1.0,
                                  jnp.inf,
                                  (mu * k - mu ** 3) / (1.0 - mu ** 2))
        return inverse_softplus(initial_kappa)

    def search_to_natural(self, search_parameters: RealArray) -> VonMisesFisherNP:
        kappa = softplus(search_parameters)
        mu = jnp.linalg.norm(self.mean, 2, axis=-1)
        q = self.mean * (kappa / mu)
        return VonMisesFisherNP(q)

    def search_gradient(self, search_parameters: RealArray) -> RealArray:
        k = self.mean.shape[-1]
        kappa = softplus(search_parameters)
        mu = jnp.linalg.norm(self.mean, 2, axis=-1)
        return _a_k(k, kappa) - mu


# Private functions --------------------------------------------------------------------------------
def _a_k(k: RealArray, kappa: RealArray) -> RealArray:
    half_k = k * 0.5
    return ive(half_k, kappa) / ive(half_k - 1.0, kappa)
