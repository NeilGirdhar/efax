from __future__ import annotations

import math
from typing import Any

from array_api_compat import get_namespace
from jax.scipy import special as jss
from tjax import Array, JaxRealArray
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..mixins.transformed_parametrization import (TransformedExpectationParametrization,
                                                  TransformedNaturalParametrization)
from ..parameter import RealField, ScalarSupport, distribution_parameter
from ..parametrization import SimpleDistribution
from .chi_square import ChiSquareEP, ChiSquareNP


@dataclass
class ChiNP(HasEntropyNP['ChiEP'],
            TransformedNaturalParametrization[ChiSquareNP, ChiSquareEP, 'ChiEP', JaxRealArray],
            SimpleDistribution):
    """The natural parametrization of the chi distribution.

    Args:
        k_over_two_minus_one: k/2 - 1 where k is the shape parameter.
    """
    k_over_two_minus_one: JaxRealArray = distribution_parameter(ScalarSupport(
        ring=RealField(minimum=-1.0)))

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=RealField(minimum=0.0))

    @override
    @classmethod
    def expectation_parametrization_cls(cls) -> type[ChiEP]:
        return ChiEP

    @override
    @classmethod
    def base_distribution_cls(cls) -> type[ChiSquareNP]:
        return ChiSquareNP

    @override
    def base_distribution(self) -> ChiSquareNP:
        return ChiSquareNP(self.k_over_two_minus_one)

    @override
    @classmethod
    def create_expectation(cls, expectation_parametrization: ChiSquareEP) -> ChiEP:
        return ChiEP(expectation_parametrization.mean_log)

    @override
    @classmethod
    def sample_to_base_sample(cls, x: Array, **fixed_parameters: Any) -> JaxRealArray:
        xp = get_namespace(x)
        return xp.square(x)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.get_namespace(x)
        return xp.log(2.0 * x) - xp.square(x) * 0.5


@dataclass
class ChiEP(HasEntropyEP[ChiNP],
            TransformedExpectationParametrization[ChiSquareEP, ChiSquareNP, ChiNP],
            SimpleDistribution):
    """The expectation parametrization of the chi distribution.

    Args:
        mean_log: E(log(x)).
    """
    mean_log: JaxRealArray = distribution_parameter(ScalarSupport())

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=RealField(minimum=0.0))

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[ChiNP]:
        return ChiNP

    @override
    @classmethod
    def base_distribution_cls(cls) -> type[ChiSquareEP]:
        return ChiSquareEP

    @override
    def base_distribution(self) -> ChiSquareEP:
        return ChiSquareEP(self.mean_log)

    @override
    def create_natural(self, natural_parametrization: ChiSquareNP) -> ChiNP:
        return ChiNP(natural_parametrization.k_over_two_minus_one)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        q = self.to_nat()
        k_over_two = q.k_over_two_minus_one + 1.0
        return -1.0 * k_over_two + 0.5 * jss.digamma(k_over_two) + 1.5 * math.log(2.0)
