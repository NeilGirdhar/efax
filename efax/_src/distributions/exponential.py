from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from tjax import JaxRealArray, Shape
from tjax.dataclasses import dataclass

from ..conjugate_prior import HasConjugatePrior
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter
from ..samplable import Samplable
from .gamma import GammaNP

__all__ = ['ExponentialNP', 'ExponentialEP']


@dataclass
class ExponentialNP(NaturalParametrization['ExponentialEP', JaxRealArray], Samplable):
    negative_rate: JaxRealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.negative_rate.shape

    def log_normalizer(self) -> JaxRealArray:
        return -jnp.log(-self.negative_rate)

    def to_exp(self) -> ExponentialEP:
        return ExponentialEP(-1.0 / self.negative_rate)

    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape)

    def sufficient_statistics(self, x: JaxRealArray) -> ExponentialEP:
        return ExponentialEP(x)

    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        return -jax.random.exponential(key, shape) / self.negative_rate


@dataclass
class ExponentialEP(HasConjugatePrior[ExponentialNP], Samplable):
    mean: JaxRealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape

    @classmethod
    def natural_parametrization_cls(cls) -> type[ExponentialNP]:
        return ExponentialNP

    def to_nat(self) -> ExponentialNP:
        return ExponentialNP(-1.0 / self.mean)

    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.shape)

    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        return jax.random.exponential(key, shape) * self.mean

    def conjugate_prior_distribution(self, n: JaxRealArray) -> GammaNP:
        return GammaNP(-n / self.mean, n)

    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.mean
