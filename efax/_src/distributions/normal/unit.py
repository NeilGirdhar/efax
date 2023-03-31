from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from tjax import JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...interfaces.conjugate_prior import HasConjugatePrior
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import ScalarSupport, distribution_parameter
from .normal import NormalNP

__all__ = ['UnitNormalNP', 'UnitNormalEP']


@dataclass
class UnitNormalNP(HasEntropyNP,
                   NaturalParametrization['UnitNormalEP', JaxRealArray],
                   Samplable):
    """The normal distribution with unit variance.

    This is a curved exponential family.
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    def domain_support(self) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        return 0.5 * (jnp.square(self.mean) + math.log(math.pi * 2.0))

    @override
    def to_exp(self) -> UnitNormalEP:
        return UnitNormalEP(self.mean)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        # The second moment of a delta distribution at x.
        return -0.5 * jnp.square(x)

    @override
    def sufficient_statistics(self, x: JaxRealArray) -> UnitNormalEP:
        return UnitNormalEP(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        return jax.random.normal(key, shape) + self.mean


@dataclass
class UnitNormalEP(HasEntropyEP[UnitNormalNP],
                   HasConjugatePrior[UnitNormalNP],
                   Samplable):
    mean: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    def domain_support(self) -> ScalarSupport:
        return ScalarSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[UnitNormalNP]:
        return UnitNormalNP

    @override
    def to_nat(self) -> UnitNormalNP:
        return UnitNormalNP(self.mean)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        # The second moment of a normal distribution with the given mean.
        return -0.5 * (jnp.square(self.mean) + 1.0)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_nat().sample(key, shape)

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> NormalNP:
        negative_half_precision = -0.5 * n
        return NormalNP(n * self.mean, negative_half_precision)

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.mean
