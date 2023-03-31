from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from tjax import JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.samplable import Samplable
from ..parameter import ScalarSupport
from .dirichlet_common import DirichletCommonEP, DirichletCommonNP

__all__ = ['BetaNP', 'BetaEP']


@dataclass
class BetaNP(DirichletCommonNP['BetaEP'], Samplable):
    """The Beta distribution.

    The best way to interpret the parameters of the beta distribution are that an observation x in
    [0, 1] represents the Bernoulli probability that outcome 0 (out of {0, 1}) is realized.  In this
    way, the Beta class coincides with a special case of the Dirichlet class.
    """
    @override
    def domain_support(self) -> ScalarSupport:
        return ScalarSupport()

    @override
    def to_exp(self) -> BetaEP:
        return BetaEP(self._exp_helper())

    @override
    def sufficient_statistics(self, x: JaxRealArray) -> BetaEP:
        return BetaEP(jnp.stack([jnp.log(x), jnp.log(1.0 - x)], axis=-1))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        alpha = 1.0 + self.alpha_minus_one
        return jax.random.beta(key, alpha[..., 0], alpha[..., 1], shape)


@dataclass
class BetaEP(DirichletCommonEP[BetaNP]):
    @override
    def domain_support(self) -> ScalarSupport:
        return ScalarSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[BetaNP]:
        return BetaNP

    @override
    def search_to_natural(self, search_parameters: JaxRealArray) -> BetaNP:
        return BetaNP(self._transform_nat_helper(search_parameters))
