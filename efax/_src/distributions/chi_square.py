from __future__ import annotations

from typing import Type

import jax.numpy as jnp
from jax.scipy import special as jss
from tjax import RealArray, Shape
from tjax.dataclasses import dataclass

from ..exp_to_nat import ExpToNat
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter

__all__ = ['ChiSquareNP', 'ChiSquareEP']


@dataclass
class ChiSquareNP(NaturalParametrization['ChiSquareEP', RealArray]):
    """
    The chi-square distribution with k degrees of freedom is the gamma distribution with shape k/2
    and rate 1/2.
    """
    k_over_two_minus_one: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.k_over_two_minus_one.shape

    def log_normalizer(self) -> RealArray:
        k_over_two = self.k_over_two_minus_one + 1.0
        return jss.gammaln(k_over_two) - k_over_two * jnp.log(0.5)

    def to_exp(self) -> ChiSquareEP:
        k_over_two = self.k_over_two_minus_one + 1.0
        return ChiSquareEP(jss.digamma(k_over_two) - jnp.log(0.5))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return -x * 0.5

    def sufficient_statistics(self, x: RealArray) -> ChiSquareEP:
        return ChiSquareEP(jnp.log(x))


# The ExpToNat mixin can be circumvented if the inverse of the digamma function were added to JAX.
@dataclass
class ChiSquareEP(ExpToNat[ChiSquareNP, ChiSquareNP]):
    mean_log: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @classmethod
    def natural_parametrization_cls(cls) -> Type[ChiSquareNP]:
        return ChiSquareNP

    @property
    def shape(self) -> Shape:
        return self.mean_log.shape

    def expected_carrier_measure(self) -> RealArray:
        q = self.to_nat()
        k_over_two = q.k_over_two_minus_one + 1.0
        return -k_over_two

    def initial_search_parameters(self) -> ChiSquareNP:
        return ChiSquareNP(jnp.zeros(self.mean_log.shape))

    def search_to_natural(self, search_parameters: ChiSquareNP) -> ChiSquareNP:
        return search_parameters

    def search_gradient(self, search_parameters: ChiSquareNP) -> ChiSquareNP:
        return self._natural_gradient(search_parameters)
