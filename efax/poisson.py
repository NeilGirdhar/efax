from __future__ import annotations

import jax.numpy as jnp
from jax.scipy import special as jss
from tjax import RealArray, Shape, dataclass

from .conjugate_prior import HasConjugatePrior
from .gamma import GammaNP
from .natural_parametrization import NaturalParametrization
from .parameter import ScalarSupport, distribution_parameter

__all__ = ['PoissonNP', 'PoissonEP']


@dataclass
class PoissonNP(NaturalParametrization['PoissonEP']):
    log_mean: RealArray = distribution_parameter(ScalarSupport())

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


@dataclass
class PoissonEP(HasConjugatePrior[PoissonNP]):
    mean: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean.shape

    def to_nat(self) -> PoissonNP:
        return PoissonNP(jnp.log(self.mean))

    # The expected_carrier_measure is -exp(-mean) * sum over k from zero to infinity of
    #   lambda ** k * log(k!) / k! = lambda ** k * log Gamma(k+1) / Gamma(k+1)

    # Overridden methods ---------------------------------------------------------------------------
    def conjugate_prior_distribution(self, n: RealArray) -> GammaNP:
        return GammaNP(-n, n * self.mean)

    def conjugate_prior_observation(self) -> RealArray:
        return self.mean
