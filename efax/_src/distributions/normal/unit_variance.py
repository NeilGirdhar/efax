from __future__ import annotations

import math
from typing import Any, Self

import jax.random as jr
from tjax import JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...interfaces.conjugate_prior import HasConjugatePrior
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import ScalarSupport, distribution_parameter
from .normal import NormalNP


@dataclass
class UnitVarianceNormalNP(HasEntropyNP['UnitVarianceNormalEP'],
                           NaturalParametrization['UnitVarianceNormalEP', JaxRealArray],
                           Samplable):
    """The natural parametrization of the normal distribution with unit variance.

    This is a curved exponential family.

    Args:
        mean: E(x).
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.array_namespace()
        return 0.5 * (xp.square(self.mean) + math.log(math.pi * 2.0))

    @override
    def to_exp(self) -> UnitVarianceNormalEP:
        return UnitVarianceNormalEP(self.mean)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        # The second moment of a delta distribution at x.
        xp = self.array_namespace(x)
        return -0.5 * xp.square(x)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray,
                              **fixed_parameters: JaxArray
                              ) -> UnitVarianceNormalEP:
        return UnitVarianceNormalEP(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        shape = self.shape if shape is None else shape + self.shape
        return jr.normal(key, shape) + self.mean


@dataclass
class UnitVarianceNormalEP(HasEntropyEP[UnitVarianceNormalNP],
                           HasConjugatePrior,
                           Samplable):
    """The expectation parametrization of the normal distribution with unit variance.

    This is a curved exponential family.

    Args:
        mean: E(x).
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[UnitVarianceNormalNP]:
        return UnitVarianceNormalNP

    @override
    def to_nat(self) -> UnitVarianceNormalNP:
        return UnitVarianceNormalNP(self.mean)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        # The second moment of a normal distribution with the given mean.
        xp = self.array_namespace()
        return -0.5 * (xp.square(self.mean) + 1.0)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_nat().sample(key, shape)

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> NormalNP:
        negative_half_precision = -0.5 * n
        return NormalNP(n * self.mean, negative_half_precision)

    @classmethod
    @override
    def from_conjugate_prior_distribution(cls, cp: NaturalParametrization[Any, Any]
                                          ) -> tuple[Self, JaxRealArray]:
        assert isinstance(cp, NormalNP)
        n = -2.0 * cp.negative_half_precision
        mean = cp.mean_times_precision / n
        return (cls(mean), n)

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.mean
