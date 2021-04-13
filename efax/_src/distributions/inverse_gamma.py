from __future__ import annotations

from typing import Any

from tjax import JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP
from ..mixins.transformed_parametrization import (TransformedExpectationParametrization,
                                                  TransformedNaturalParametrization)
from ..parameter import ScalarSupport, distribution_parameter
from ..parametrization import SimpleDistribution
from .gamma import GammaEP, GammaNP


@dataclass
class InverseGammaNP(TransformedNaturalParametrization[GammaNP, GammaEP, 'InverseGammaEP',
                                                       JaxRealArray],
                     Samplable):
    negative_scale: JaxRealArray = distribution_parameter(ScalarSupport())
    negative_shape_minus_one: JaxRealArray = distribution_parameter(ScalarSupport())

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    @classmethod
    def base_distribution_cls(cls) -> type[GammaNP]:
        return GammaNP

    @override
    def base_distribution(self) -> GammaNP:
        return GammaNP(self.negative_scale, -self.negative_shape_minus_one - 2.0)

    @override
    @classmethod
    def create_expectation(cls, expectation_parametrization: GammaEP) -> InverseGammaEP:
        return InverseGammaEP(expectation_parametrization.mean,
                              expectation_parametrization.mean_log)

    @override
    @classmethod
    def sample_to_base_sample(cls, x: JaxRealArray, **fixed_parameters: Any) -> JaxRealArray:
        return 1.0 / x

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        y = self.base_distribution().sample(key, shape)
        return 1.0 / y  # Convert base sample to sample.

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.get_namespace()
        return xp.zeros(x.shape)


@dataclass
class InverseGammaEP(TransformedExpectationParametrization[GammaEP, GammaNP, InverseGammaNP],
                     HasEntropyEP[InverseGammaNP],
                     SimpleDistribution):
    mean_reciprocal: JaxRealArray = distribution_parameter(ScalarSupport())
    negative_mean_log: JaxRealArray = distribution_parameter(ScalarSupport())

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    @classmethod
    def base_distribution_cls(cls) -> type[GammaEP]:
        return GammaEP

    @override
    def base_distribution(self) -> GammaEP:
        return GammaEP(self.mean_reciprocal, self.negative_mean_log)

    @override
    def create_natural(self, natural_parametrization: GammaNP) -> InverseGammaNP:
        return InverseGammaNP(natural_parametrization.negative_rate,
                              -natural_parametrization.shape_minus_one - 2.0)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.get_namespace()
        return xp.zeros(self.shape)
