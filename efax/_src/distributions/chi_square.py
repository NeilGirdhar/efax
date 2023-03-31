from __future__ import annotations

import jax.numpy as jnp
from jax.random import KeyArray, chisquare
from jax.scipy import special as jss
from tjax import JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.samplable import Samplable
from ..mixins.exp_to_nat import ExpToNat
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter

__all__ = ['ChiSquareNP', 'ChiSquareEP']


@dataclass
class ChiSquareNP(HasEntropyNP, NaturalParametrization['ChiSquareEP', JaxRealArray], Samplable):
    """The chi-square distribution with k degrees of freedom.

    This is the gamma distribution with shape k/2 and rate 1/2.
    """
    k_over_two_minus_one: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.k_over_two_minus_one.shape

    @override
    def domain_support(self) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        k_over_two = self.k_over_two_minus_one + 1.0
        return jss.gammaln(k_over_two) - k_over_two * jnp.log(0.5)

    @override
    def to_exp(self) -> ChiSquareEP:
        k_over_two = self.k_over_two_minus_one + 1.0
        return ChiSquareEP(jss.digamma(k_over_two) - jnp.log(0.5))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return -x * 0.5

    @override
    def sufficient_statistics(self, x: JaxRealArray) -> ChiSquareEP:
        return ChiSquareEP(jnp.log(x))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        degrees_of_freedom = (self.k_over_two_minus_one + 1.0) * 2.0
        return chisquare(key, degrees_of_freedom, shape)


# The ExpToNat mixin can be circumvented if the inverse of the digamma function were added to JAX.
@dataclass
class ChiSquareEP(HasEntropyEP[ChiSquareNP], ExpToNat[ChiSquareNP, ChiSquareNP]):
    mean_log: JaxRealArray = distribution_parameter(ScalarSupport())

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[ChiSquareNP]:
        return ChiSquareNP

    @property
    @override
    def shape(self) -> Shape:
        return self.mean_log.shape

    @override
    def domain_support(self) -> ScalarSupport:
        return ScalarSupport()

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        q = self.to_nat()
        k_over_two = q.k_over_two_minus_one + 1.0
        return -k_over_two

    @override
    def initial_search_parameters(self) -> ChiSquareNP:
        return ChiSquareNP(jnp.zeros(self.mean_log.shape))

    @override
    def search_to_natural(self, search_parameters: ChiSquareNP) -> ChiSquareNP:
        return search_parameters

    @override
    def search_gradient(self, search_parameters: ChiSquareNP) -> ChiSquareNP:
        return self._natural_gradient(search_parameters)
