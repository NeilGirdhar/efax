from __future__ import annotations

import math
from typing import Any, Self

import jax
from tjax import JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...expectation_parametrization import ExpectationParametrization
from ...interfaces.conjugate_prior import HasGeneralizedConjugatePrior
from ...interfaces.multidimensional import Multidimensional
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import VectorSupport, distribution_parameter
from .diagonal import MultivariateDiagonalNormalNP
from .isotropic import IsotropicNormalNP


@dataclass
class MultivariateUnitNormalNP(HasEntropyNP['MultivariateUnitNormalEP'],
                               NaturalParametrization['MultivariateUnitNormalEP', JaxRealArray],
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
        xp = self.get_namespace()
        return 0.5 * (xp.sum(xp.square(self.mean), axis=-1)
                      + self.dimensions() * math.log(math.pi * 2.0))

    @override
    def to_exp(self) -> MultivariateUnitNormalEP:
        return MultivariateUnitNormalEP(self.mean)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        # The second moment of a delta distribution at x.
        xp = self.get_namespace(x)
        return -0.5 * xp.sum(xp.square(x), axis=-1)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> MultivariateUnitNormalEP:
        return MultivariateUnitNormalEP(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        return jax.random.normal(key, shape) + self.mean

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]


@dataclass
class MultivariateUnitNormalEP(
        HasEntropyEP[MultivariateUnitNormalNP],
        HasGeneralizedConjugatePrior,
        Samplable,
        ExpectationParametrization[MultivariateUnitNormalNP],
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
    def natural_parametrization_cls(cls) -> type[MultivariateUnitNormalNP]:
        return MultivariateUnitNormalNP

    @override
    def to_nat(self) -> MultivariateUnitNormalNP:
        return MultivariateUnitNormalNP(self.mean)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        # The second moment of a normal distribution with the given mean.
        xp = self.get_namespace()
        return -0.5 * (xp.sum(xp.square(self.mean), axis=-1) + self.dimensions())

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_nat().sample(key, shape)

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> IsotropicNormalNP:
        xp = self.get_namespace()
        negative_half_precision = -0.5 * n
        return IsotropicNormalNP(n[..., xp.newaxis] * self.mean, negative_half_precision)

    @classmethod
    @override
    def from_conjugate_prior_distribution(cls, cp: NaturalParametrization[Any, Any]
                                          ) -> tuple[Self, JaxRealArray]:
        assert isinstance(cp, IsotropicNormalNP)
        xp = cp.get_namespace()
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
