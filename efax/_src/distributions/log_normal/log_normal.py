from __future__ import annotations

import jax.random as jr
from array_api_compat import array_namespace
from tjax import Array, JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...mixins.transformed_parametrization import (TransformedExpectationParametrization,
                                                   TransformedNaturalParametrization)
from ...parameter import ScalarSupport, distribution_parameter, negative_support, positive_support
from ..normal.normal import NormalEP, NormalNP


@dataclass
class LogNormalNP(Samplable,
                  HasEntropyNP['LogNormalEP'],
                  TransformedNaturalParametrization[NormalNP, NormalEP, 'LogNormalEP',
                                                    JaxRealArray]):
    """The natural parametrization of the LogNormal distribution.

    Args:
        eta: -1 / (2 * sigma^2).
    """
    mean_times_precision: JaxRealArray = distribution_parameter(ScalarSupport())
    negative_half_precision: JaxRealArray = distribution_parameter(ScalarSupport(
        ring=negative_support))

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    @classmethod
    def expectation_parametrization_cls(cls) -> type[LogNormalEP]:
        return LogNormalEP

    @override
    @classmethod
    def base_distribution_cls(cls) -> type[NormalNP]:
        return NormalNP

    @override
    def base_distribution(self) -> NormalNP:
        return NormalNP(self.mean_times_precision, self.negative_half_precision)

    @override
    @classmethod
    def create_expectation_from_base(cls, expectation_parametrization: NormalEP) -> LogNormalEP:
        return LogNormalEP(expectation_parametrization.mean,
                           expectation_parametrization.second_moment)

    @override
    @classmethod
    def sample_to_base_sample(cls, x: Array, **fixed_parameters: JaxArray) -> JaxRealArray:
        xp = array_namespace(x)
        return xp.log(x)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return -xp.log(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        shape = self.shape if shape is None else shape + self.shape
        xp = self.array_namespace()
        normal_dp = self.base_distribution().to_deviation_parametrization()
        return jr.lognormal(key, normal_dp.deviation, shape) * xp.exp(normal_dp.mean)


@dataclass
class LogNormalEP(Samplable,
                  HasEntropyEP[LogNormalNP],
                  TransformedExpectationParametrization[NormalEP, NormalNP, LogNormalNP]):
    """The expectation parametrization of the LogNormal distribution.

    Args:
        chi: 2 * sigma^2.
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport())
    second_moment: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[LogNormalNP]:
        return LogNormalNP

    @override
    @classmethod
    def base_distribution_cls(cls) -> type[NormalEP]:
        return NormalEP

    @override
    def base_distribution(self) -> NormalEP:
        return NormalEP(self.mean, self.second_moment)

    @override
    @classmethod
    def create_natural_from_base(cls, natural_parametrization: NormalNP) -> LogNormalNP:
        return LogNormalNP(natural_parametrization.mean_times_precision,
                           natural_parametrization.negative_half_precision)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return -self.mean

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        shape = self.shape if shape is None else shape + self.shape
        xp = self.array_namespace()
        normal_dp = self.base_distribution().to_deviation_parametrization()
        return jr.lognormal(key, normal_dp.deviation, shape) * xp.exp(normal_dp.mean)
