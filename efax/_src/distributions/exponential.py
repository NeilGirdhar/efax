from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from tjax import JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.conjugate_prior import HasConjugatePrior
from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter
from .gamma import GammaNP

__all__ = ['ExponentialNP', 'ExponentialEP']


@dataclass
class ExponentialNP(HasEntropyNP, NaturalParametrization['ExponentialEP', JaxRealArray], Samplable):
    negative_rate: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.negative_rate.shape

    @override
    def domain_support(self) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        return -jnp.log(-self.negative_rate)

    @override
    def to_exp(self) -> ExponentialEP:
        return ExponentialEP(-1.0 / self.negative_rate)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape)

    @override
    def sufficient_statistics(self, x: JaxRealArray) -> ExponentialEP:
        return ExponentialEP(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        return -jax.random.exponential(key, shape) / self.negative_rate


@dataclass
class ExponentialEP(HasEntropyEP[ExponentialNP], HasConjugatePrior[ExponentialNP], Samplable):
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
    def natural_parametrization_cls(cls) -> type[ExponentialNP]:
        return ExponentialNP

    @override
    def to_nat(self) -> ExponentialNP:
        return ExponentialNP(-1.0 / self.mean)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        return jax.random.exponential(key, shape) * self.mean

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> GammaNP:
        return GammaNP(-n / self.mean, n)

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.mean
