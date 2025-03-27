from __future__ import annotations

from array_api_compat import array_namespace
from tjax import Array, JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..mixins.transformed_parametrization import (TransformedExpectationParametrization,
                                                  TransformedNaturalParametrization)
from ..parameter import RealField, ScalarSupport, distribution_parameter, negative_support
from .gamma import GammaEP, GammaNP


@dataclass
class InverseGammaNP(Samplable,
                     HasEntropyNP['InverseGammaEP'],
                     TransformedNaturalParametrization[GammaNP, GammaEP, 'InverseGammaEP',
                                                       JaxRealArray]):
    """The natural parametrization of the InverseGamma distribution."""
    negative_scale: JaxRealArray = distribution_parameter(ScalarSupport(ring=negative_support))
    shape_minus_one: JaxRealArray = distribution_parameter(ScalarSupport(
        ring=RealField(minimum=-1.0)))

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    @classmethod
    def expectation_parametrization_cls(cls) -> type[InverseGammaEP]:
        return InverseGammaEP

    @override
    @classmethod
    def base_distribution_cls(cls) -> type[GammaNP]:
        return GammaNP

    @override
    def base_distribution(self) -> GammaNP:
        return GammaNP(self.negative_scale, self.shape_minus_one)

    @override
    @classmethod
    def create_expectation_from_base(cls, expectation_parametrization: GammaEP) -> InverseGammaEP:
        return InverseGammaEP(expectation_parametrization.mean,
                              expectation_parametrization.mean_log)

    @override
    @classmethod
    def sample_to_base_sample(cls, x: Array, **fixed_parameters: JaxArray) -> JaxRealArray:
        xp = array_namespace(x)
        return xp.reciprocal(x)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return -2.0 * xp.log(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.reciprocal(self.base_distribution().sample(key, shape))


@dataclass
class InverseGammaEP(Samplable,
                     HasEntropyEP[InverseGammaNP],
                     TransformedExpectationParametrization[GammaEP, GammaNP, InverseGammaNP]):
    """The expectation parametrization of the InverseGamma distribution."""
    base_mean: JaxRealArray = distribution_parameter(ScalarSupport())
    base_mean_log: JaxRealArray = distribution_parameter(ScalarSupport())

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[InverseGammaNP]:
        return InverseGammaNP

    @override
    @classmethod
    def base_distribution_cls(cls) -> type[GammaEP]:
        return GammaEP

    @override
    def base_distribution(self) -> GammaEP:
        return GammaEP(self.base_mean, self.base_mean_log)

    @override
    @classmethod
    def create_natural_from_base(cls, natural_parametrization: GammaNP) -> InverseGammaNP:
        return InverseGammaNP(natural_parametrization.negative_rate,
                              natural_parametrization.shape_minus_one)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return 2.0 * self.base_mean_log

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.reciprocal(self.base_distribution().sample(key, shape))
