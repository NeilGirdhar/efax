from __future__ import annotations

import math

import jax.numpy as jnp
from jax.nn import softplus
from tjax import RealArray, RealNumeric, Shape, inverse_softplus
from tjax.dataclasses import dataclass

from ..exp_to_nat import ExpToNat
from ..multidimensional import Multidimensional
from ..natural_parametrization import NaturalParametrization
from ..parameter import VectorSupport, distribution_parameter
from ..tools import iv_ratio, log_ive

__all__ = ['VonMisesFisherNP', 'VonMisesFisherEP']


@dataclass
class VonMisesFisherNP(NaturalParametrization['VonMisesFisherEP', RealArray],
                       Multidimensional):
    mean_times_concentration: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean_times_concentration.shape[:-1]

    def log_normalizer(self) -> RealArray:
        half_k = self.dimensions() * 0.5
        kappa = jnp.linalg.norm(self.mean_times_concentration, 2, axis=-1)
        return (kappa
                - (half_k - 1.0) * jnp.log(kappa)
                + half_k * jnp.log(2.0 * math.pi)
                + log_ive(half_k - 1.0, kappa))

    def to_exp(self) -> VonMisesFisherEP:
        q = self.mean_times_concentration
        kappa: RealArray = jnp.linalg.norm(q, 2, axis=-1, keepdims=True)
        return VonMisesFisherEP(
            jnp.where(kappa == 0.0,
                      q,
                      q * (_a_k(self.dimensions(), kappa) / kappa)))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(self.shape)

    def sufficient_statistics(self, x: RealArray) -> VonMisesFisherEP:
        return VonMisesFisherEP(x)

    def dimensions(self) -> int:
        return self.mean_times_concentration.shape[-1]

    # New methods ----------------------------------------------------------------------------------
    def to_kappa_angle(self) -> tuple[RealArray, RealArray]:
        if self.dimensions() != 2:
            raise ValueError
        kappa: RealArray = jnp.linalg.norm(self.mean_times_concentration, axis=-1)
        angle = jnp.where(kappa == 0.0,
                          0.0,
                          jnp.arctan2(self.mean_times_concentration[..., 1],
                                      self.mean_times_concentration[..., 0]))
        return kappa, angle


@dataclass
class VonMisesFisherEP(ExpToNat[VonMisesFisherNP, RealArray], Multidimensional):
    mean: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @classmethod
    def natural_parametrization_cls(cls) -> type[VonMisesFisherNP]:
        return VonMisesFisherNP

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape)

    def initial_search_parameters(self) -> RealArray:
        mu: RealArray = jnp.linalg.norm(self.mean, 2, axis=-1)
        # 0 <= mu <= 1.0
        initial_kappa = jnp.where(mu == 1.0,
                                  jnp.inf,
                                  (mu * self.dimensions() - mu ** 3) / (1.0 - mu ** 2))
        return inverse_softplus(initial_kappa)

    def search_to_natural(self, search_parameters: RealArray) -> VonMisesFisherNP:
        kappa = softplus(search_parameters)
        mu = jnp.linalg.norm(self.mean, 2, axis=-1)
        q = self.mean * (kappa / mu)[..., jnp.newaxis]
        return VonMisesFisherNP(q)

    def search_gradient(self, search_parameters: RealArray) -> RealArray:
        kappa = softplus(search_parameters)
        mu = jnp.linalg.norm(self.mean, 2, axis=-1)
        return _a_k(self.dimensions(), kappa) - mu

    def dimensions(self) -> int:
        return self.mean.shape[-1]


# Private functions --------------------------------------------------------------------------------
def _a_k(k: RealNumeric, kappa: RealNumeric) -> RealArray:
    return iv_ratio(k * 0.5, kappa)
