from __future__ import annotations

import math

import jax.random as jr
import jax.scipy.special as jss
from array_api_compat import array_namespace
from tjax import JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.samplable import Samplable
from ..mixins.exp_to_nat.exp_to_nat import ExpToNat
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import RealField, ScalarSupport, distribution_parameter
from ..parametrization import SimpleDistribution


@dataclass
class ChiSquareNP(HasEntropyNP['ChiSquareEP'],
                  Samplable,
                  NaturalParametrization['ChiSquareEP', JaxRealArray]):
    """The natural parameters of the chi-square distribution with k degrees of freedom.

    This is the gamma distribution with shape k/2 and rate 1/2.

    Args:
        k_over_two_minus_one: k/2 - 1.
    """
    k_over_two_minus_one: JaxRealArray = distribution_parameter(ScalarSupport(
        ring=RealField(minimum=-1.0)))

    @property
    @override
    def shape(self) -> Shape:
        return self.k_over_two_minus_one.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        k_over_two = self.k_over_two_minus_one + 1.0
        return jss.gammaln(k_over_two) - k_over_two * math.log(0.5)

    @override
    def to_exp(self) -> ChiSquareEP:
        k_over_two = self.k_over_two_minus_one + 1.0
        return ChiSquareEP(jss.digamma(k_over_two) - math.log(0.5))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return -x * 0.5

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray
                              ) -> ChiSquareEP:
        xp = array_namespace(x)
        return ChiSquareEP(xp.log(x))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        degrees_of_freedom = (self.k_over_two_minus_one + 1.0) * 2.0
        return jr.chisquare(key, degrees_of_freedom, shape)


# The ExpToNat mixin can be circumvented if the inverse of the digamma function were added to JAX.
@dataclass
class ChiSquareEP(HasEntropyEP[ChiSquareNP],
                  Samplable,
                  ExpToNat[ChiSquareNP],
                  SimpleDistribution):
    """The expectation parameters of the chi-square distribution with k degrees of freedom.

    This is the gamma distribution with shape k/2 and rate 1/2.

    Args:
        mean_log: E(log(x)).
    """
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
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        q = self.to_nat()
        k_over_two = q.k_over_two_minus_one + 1.0
        return -k_over_two

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_nat().sample(key, shape)
