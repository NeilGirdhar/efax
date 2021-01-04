from __future__ import annotations

from typing import Type

from jax import numpy as jnp
from jax.scipy import special as jss
from tjax import RealArray, Shape, dataclass

from .exp_to_nat import ExpToNat
from .natural_parametrization import NaturalParametrization
from .parameter import distribution_parameter

__all__ = ['ChiSquareNP', 'ChiSquareEP']


@dataclass
class ChiSquareNP(NaturalParametrization['ChiSquareEP']):
    """
    The chi-square distribution with k degrees of freedom is the gamma distribution with shape k/2
    and rate 1/2.
    """
    k_over_two_minus_one: RealArray = distribution_parameter(axes=0)

    # Implemented methods --------------------------------------------------------------------------
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


@dataclass
class ChiSquareEP(ExpToNat[ChiSquareNP]):
    mean_log: RealArray = distribution_parameter(axes=0)

    # Implemented methods --------------------------------------------------------------------------
    @classmethod
    def natural_parametrization_cls(cls) -> Type[ChiSquareNP]:
        return ChiSquareNP

    def shape(self) -> Shape:
        return self.mean_log.shape

    # The expected_carrier_measure should return k/2, which requires the inverse of the digamma
    # function.

    @classmethod
    def transform_natural_for_iteration(cls, iteration_natural: ChiSquareNP) -> ChiSquareNP:
        return iteration_natural
