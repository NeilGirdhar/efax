from __future__ import annotations

from typing import Optional, Type

import jax
import jax.numpy as jnp
from tjax import Generator, RealArray, Shape
from tjax.dataclasses import dataclass

from ..conjugate_prior import HasConjugatePrior
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter
from ..samplable import Samplable
from .gamma import GammaNP

__all__ = ['ExponentialNP', 'ExponentialEP']


@dataclass
class ExponentialNP(NaturalParametrization['ExponentialEP', RealArray], Samplable):
    negative_rate: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.negative_rate.shape

    def log_normalizer(self) -> RealArray:
        return -jnp.log(-self.negative_rate)

    def to_exp(self) -> ExponentialEP:
        return ExponentialEP(-1.0 / self.negative_rate)

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape)

    def sufficient_statistics(self, x: RealArray) -> ExponentialEP:
        return ExponentialEP(x)

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> RealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        return -jax.random.exponential(rng.key, shape) / self.negative_rate


@dataclass
class ExponentialEP(HasConjugatePrior[ExponentialNP], Samplable):
    mean: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape

    @classmethod
    def natural_parametrization_cls(cls) -> Type[ExponentialNP]:
        return ExponentialNP

    def to_nat(self) -> ExponentialNP:
        return ExponentialNP(-1.0 / self.mean)

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape)

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> RealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        return jax.random.exponential(rng.key, shape) * self.mean

    # Overridden methods ---------------------------------------------------------------------------
    def conjugate_prior_distribution(self, n: RealArray) -> GammaNP:
        return GammaNP(-n / self.mean, n)

    def conjugate_prior_observation(self) -> RealArray:
        return self.mean
