from __future__ import annotations

from typing import Iterable

from jax import numpy as jnp
from tjax import RealArray, Shape, dataclass

from .conjugate_prior import HasConjugatePrior
from .exponential_family import NaturalParametrization
from .gamma import GammaNP

__all__ = ['ExponentialNP', 'ExponentialEP']


@dataclass
class ExponentialNP(NaturalParametrization['ExponentialEP']):
    negative_rate: RealArray

    # Implemented methods --------------------------------------------------------------------------
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

    @classmethod
    def field_axes(cls) -> Iterable[int]:
        yield 0


@dataclass
class ExponentialEP(HasConjugatePrior[ExponentialNP]):
    mean: RealArray

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean.shape

    def to_nat(self) -> ExponentialNP:
        return ExponentialNP(-1.0 / self.mean)

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape())

    # Overridden methods ---------------------------------------------------------------------------
    def conjugate_prior_distribution(self, n: RealArray) -> GammaNP:
        return GammaNP(-n / self.mean, n)

    def conjugate_prior_observation(self) -> RealArray:
        return self.mean
