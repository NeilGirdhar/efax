from __future__ import annotations

from typing import cast

import jax.random as jr
from tjax import Array, JaxArray, JaxRealArray, KeyArray, Shape, inverse_softplus, softplus
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...interfaces.samplable import Samplable
from ...mixins.transformed_parametrization import (TransformedExpectationParametrization,
                                                   TransformedNaturalParametrization)
from ...parameter import ScalarSupport, distribution_parameter, negative_support, positive_support
from ..normal.normal import NormalEP, NormalNP


@dataclass
class SoftplusNormalNP(
        Samplable,
        TransformedNaturalParametrization[NormalNP, NormalEP, 'SoftplusNormalEP', JaxRealArray]):
    """The natural parametrization of the softplus-normal distribution."""
    mean_times_precision: JaxRealArray = distribution_parameter(ScalarSupport())
    negative_half_precision: JaxRealArray = distribution_parameter(ScalarSupport(
        ring=negative_support))

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    @classmethod
    def expectation_parametrization_cls(cls) -> type[SoftplusNormalEP]:
        return SoftplusNormalEP

    @override
    @classmethod
    def base_distribution_cls(cls) -> type[NormalNP]:
        return NormalNP

    @override
    def base_distribution(self) -> NormalNP:
        return NormalNP(self.mean_times_precision, self.negative_half_precision)

    @override
    @classmethod
    def create_expectation_from_base(cls, expectation_parametrization: NormalEP
                                     ) -> SoftplusNormalEP:
        return SoftplusNormalEP(expectation_parametrization.mean,
                                expectation_parametrization.second_moment)

    @override
    @classmethod
    def sample_to_base_sample(cls, x: Array, **fixed_parameters: JaxArray) -> JaxRealArray:
        return cast('JaxRealArray', inverse_softplus(x))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return -xp.log1p(-xp.exp(-x))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        shape = self.shape if shape is None else shape + self.shape
        normal_dp = self.base_distribution().to_deviation_parametrization()
        return softplus(jr.normal(key, shape) * normal_dp.deviation + normal_dp.mean)


@dataclass
class SoftplusNormalEP(Samplable,
                       TransformedExpectationParametrization[NormalEP, NormalNP, SoftplusNormalNP]):
    """The expectation parametrization of the softplus-normal distribution."""
    mean: JaxRealArray = distribution_parameter(ScalarSupport())
    second_moment: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[SoftplusNormalNP]:
        return SoftplusNormalNP

    @override
    @classmethod
    def base_distribution_cls(cls) -> type[NormalEP]:
        return NormalEP

    @override
    def base_distribution(self) -> NormalEP:
        return NormalEP(self.mean, self.second_moment)

    @override
    @classmethod
    def create_natural_from_base(cls, natural_parametrization: NormalNP) -> SoftplusNormalNP:
        return SoftplusNormalNP(natural_parametrization.mean_times_precision,
                                natural_parametrization.negative_half_precision)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        shape = self.shape if shape is None else shape + self.shape
        normal_dp = self.base_distribution().to_deviation_parametrization()
        return softplus(jr.normal(key, shape) * normal_dp.deviation + normal_dp.mean)
