from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from tjax import RealArray, Shape
from tjax.dataclasses import dataclass

from ..samplable import Samplable
from .dirichlet_common import DirichletCommonEP, DirichletCommonNP

__all__ = ['BetaNP', 'BetaEP']


@dataclass
class BetaNP(DirichletCommonNP['BetaEP'], Samplable):
    """The Beta distribution.

    The best way to interpret the parameters of the beta distribution are that an observation x in
    [0, 1] represents the Bernoulli probability that outcome 0 (out of {0, 1}) is realized.  In this
    way, the Beta class coincides with a special case of the Dirichlet class.
    """
    # Overridden methods ---------------------------------------------------------------------------
    def to_exp(self) -> BetaEP:
        return BetaEP(self._exp_helper())

    def sufficient_statistics(self, x: RealArray) -> BetaEP:
        return BetaEP(jnp.stack([jnp.log(x), jnp.log(1.0 - x)], axis=-1))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape)

    def sample(self, key: KeyArray, shape: Shape | None = None) -> RealArray:
        if shape is not None:
            shape += self.shape
        return jax.random.beta(key,
                               1.0 + self.alpha_minus_one[..., 0],
                               1.0 + self.alpha_minus_one[..., 1],
                               shape)


@dataclass
class BetaEP(DirichletCommonEP[BetaNP]):
    # Implemented methods --------------------------------------------------------------------------
    @classmethod
    def natural_parametrization_cls(cls) -> type[BetaNP]:
        return BetaNP

    def search_to_natural(self, search_parameters: RealArray) -> BetaNP:
        return BetaNP(self._transform_nat_helper(search_parameters))
