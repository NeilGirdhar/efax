from __future__ import annotations

from typing import Any, Self, cast

import jax.random as jr
from tjax import Array, JaxArray, JaxRealArray, KeyArray, Shape, inverse_softplus, softplus
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...interfaces.conjugate_prior import HasConjugatePrior
from ...interfaces.samplable import Samplable
from ...mixins.transformed_parametrization import (TransformedExpectationParametrization,
                                                   TransformedNaturalParametrization)
from ...natural_parametrization import NaturalParametrization
from ...parameter import ScalarSupport, distribution_parameter
from ..normal.normal import NormalNP
from ..normal.unit_variance import UnitVarianceNormalEP, UnitVarianceNormalNP


@dataclass
class UnitVarianceSoftplusNormalNP(
        Samplable,
        TransformedNaturalParametrization[UnitVarianceNormalNP, UnitVarianceNormalEP,
                                          'UnitVarianceSoftplusNormalEP',
                                          JaxRealArray]):
    """natural parametrization of the softplus-normal distribution with unit variance."""
    mean: JaxRealArray = distribution_parameter(ScalarSupport())

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    @classmethod
    def expectation_parametrization_cls(cls) -> type[UnitVarianceSoftplusNormalEP]:
        return UnitVarianceSoftplusNormalEP

    @override
    @classmethod
    def base_distribution_cls(cls) -> type[UnitVarianceNormalNP]:
        return UnitVarianceNormalNP

    @override
    def base_distribution(self) -> UnitVarianceNormalNP:
        return UnitVarianceNormalNP(self.mean)

    @override
    @classmethod
    def create_expectation_from_base(cls, expectation_parametrization: UnitVarianceNormalEP
                                     ) -> UnitVarianceSoftplusNormalEP:
        return UnitVarianceSoftplusNormalEP(expectation_parametrization.mean)

    @override
    @classmethod
    def sample_to_base_sample(cls, x: Array, **fixed_parameters: JaxArray) -> JaxRealArray:
        return cast('JaxRealArray', inverse_softplus(x))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return -xp.log1p(-xp.exp(-x)) - 0.5 * xp.square(inverse_softplus(x))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        shape = self.shape if shape is None else shape + self.shape
        return softplus(jr.normal(key, shape) + self.mean)


@dataclass
class UnitVarianceSoftplusNormalEP(
        HasConjugatePrior,
        Samplable,
        TransformedExpectationParametrization[UnitVarianceNormalEP, UnitVarianceNormalNP,
                                              UnitVarianceSoftplusNormalNP]):
    """The expectation parametrization of the softplus-normal distribution with unit variance."""
    mean: JaxRealArray = distribution_parameter(ScalarSupport())

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[UnitVarianceSoftplusNormalNP]:
        return UnitVarianceSoftplusNormalNP

    @override
    @classmethod
    def base_distribution_cls(cls) -> type[UnitVarianceNormalEP]:
        return UnitVarianceNormalEP

    @override
    def base_distribution(self) -> UnitVarianceNormalEP:
        return UnitVarianceNormalEP(self.mean)

    @override
    @classmethod
    def create_natural_from_base(cls, natural_parametrization: UnitVarianceNormalNP
                                 ) -> UnitVarianceSoftplusNormalNP:
        return UnitVarianceSoftplusNormalNP(natural_parametrization.mean)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        shape = self.shape if shape is None else shape + self.shape
        return softplus(jr.normal(key, shape) + self.mean)

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> NormalNP:
        normal_np = self.base_distribution().conjugate_prior_distribution(n)
        return NormalNP(normal_np.mean_times_precision, normal_np.negative_half_precision)

    @classmethod
    @override
    def from_conjugate_prior_distribution(cls, cp: NaturalParametrization[Any, Any]
                                          ) -> tuple[Self, JaxRealArray]:
        uvn, n = UnitVarianceNormalEP.from_conjugate_prior_distribution(cp)
        return (cls(uvn.mean), n)

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.mean
