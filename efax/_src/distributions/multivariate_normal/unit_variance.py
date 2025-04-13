from __future__ import annotations

import math
from typing import Any, Self

import jax.random as jr
from tjax import JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...interfaces.conjugate_prior import HasGeneralizedConjugatePrior
from ...interfaces.multidimensional import Multidimensional
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import VectorSupport, distribution_parameter
from .diagonal import MultivariateDiagonalNormalNP
from .isotropic import IsotropicNormalNP


@dataclass
class MultivariateUnitVarianceNormalNP(
        HasEntropyNP['MultivariateUnitVarianceNormalEP'],
        NaturalParametrization['MultivariateUnitVarianceNormalEP', JaxRealArray],
        Multidimensional,
        Samplable):
    """The natural parametrization of the multivariate normal distribution with unit variance.

    This is a curved exponential family.

    Args:
        mean: E(x).
    """
    mean: JaxRealArray = distribution_parameter(VectorSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.array_namespace()
        return 0.5 * (xp.sum(xp.square(self.mean), axis=-1)
                      + self.dimensions() * math.log(math.pi * 2.0))

    @override
    def to_exp(self) -> MultivariateUnitVarianceNormalEP:
        return MultivariateUnitVarianceNormalEP(self.mean)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        # The second moment of a delta distribution at x.
        xp = self.array_namespace(x)
        return -0.5 * xp.sum(xp.square(x), axis=-1)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray
                              ) -> MultivariateUnitVarianceNormalEP:
        return MultivariateUnitVarianceNormalEP(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        shape = self.mean.shape if shape is None else shape + self.mean.shape
        return jr.normal(key, shape) + self.mean

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]


@dataclass
class MultivariateUnitVarianceNormalEP(
        HasEntropyEP[MultivariateUnitVarianceNormalNP],
        HasGeneralizedConjugatePrior,
        Samplable,
        Multidimensional):
    """The expectation parametrization of the multivariate normal distribution with unit variance.

    This is a curved exponential family.

    Args:
        mean: E(x).
    """
    mean: JaxRealArray = distribution_parameter(VectorSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[MultivariateUnitVarianceNormalNP]:
        return MultivariateUnitVarianceNormalNP

    @override
    def to_nat(self) -> MultivariateUnitVarianceNormalNP:
        return MultivariateUnitVarianceNormalNP(self.mean)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        # The second moment of a normal distribution with the given mean.
        xp = self.array_namespace()
        return -0.5 * (xp.sum(xp.square(self.mean), axis=-1) + self.dimensions())

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_nat().sample(key, shape)

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> IsotropicNormalNP:
        xp = self.array_namespace()
        negative_half_precision = -0.5 * n
        return IsotropicNormalNP(n[..., xp.newaxis] * self.mean, negative_half_precision)

    @classmethod
    @override
    def from_conjugate_prior_distribution(cls, cp: NaturalParametrization[Any, Any]
                                          ) -> tuple[Self, JaxRealArray]:
        assert isinstance(cp, IsotropicNormalNP)
        xp = cp.array_namespace()
        n = -2.0 * cp.negative_half_precision
        mean = cp.mean_times_precision / n[..., xp.newaxis]
        return (cls(mean), n)

    @override
    def generalized_conjugate_prior_distribution(self, n: JaxRealArray
                                                 ) -> MultivariateDiagonalNormalNP:
        negative_half_precision = -0.5 * n
        return MultivariateDiagonalNormalNP(n * self.mean, negative_half_precision)

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.mean

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]
