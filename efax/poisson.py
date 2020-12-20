from __future__ import annotations

from typing import Iterable

from jax import numpy as jnp
from jax.scipy import special as jss
from tjax import RealArray, Shape, dataclass

from .conjugate_prior import HasConjugatePrior
from .exponential_family import NaturalParametrization
from .gamma import GammaNP

__all__ = ['PoissonNP', 'PoissonEP']


@dataclass
class PoissonNP(NaturalParametrization['PoissonEP']):
    log_mean: RealArray

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.log_mean.shape

    def log_normalizer(self) -> RealArray:
        return jnp.exp(self.log_mean)

    def to_exp(self) -> PoissonEP:
        return PoissonEP(jnp.exp(self.log_mean))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return -jss.gammaln(x + 1)

    def sufficient_statistics(self, x: RealArray) -> PoissonEP:
        return PoissonEP(x)

    @classmethod
    def field_axes(cls) -> Iterable[int]:
        yield 0


@dataclass
class PoissonEP(HasConjugatePrior[PoissonNP]):
    mean: RealArray

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean.shape

    def to_nat(self) -> PoissonNP:
        return PoissonNP(jnp.log(self.mean))

    # The expected_carrier_measure is unknown.

    # Overridden methods ---------------------------------------------------------------------------
    def conjugate_prior_distribution(self, n: RealArray) -> GammaNP:
        return GammaNP(-n, n * self.mean)

    def conjugate_prior_observation(self) -> RealArray:
        return self.mean
