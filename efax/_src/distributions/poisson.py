from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from jax.scipy import special as jss
from tjax import JaxRealArray, Shape
from tjax.dataclasses import dataclass

from ..conjugate_prior import HasConjugatePrior
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter
from ..samplable import Samplable
from .gamma import GammaNP

__all__ = ['PoissonNP', 'PoissonEP']


@dataclass
class PoissonNP(NaturalParametrization['PoissonEP', JaxRealArray]):
    log_mean: JaxRealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.log_mean.shape

    def log_normalizer(self) -> JaxRealArray:
        return jnp.exp(self.log_mean)

    def to_exp(self) -> PoissonEP:
        return PoissonEP(jnp.exp(self.log_mean))

    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return -jss.gammaln(x + 1)

    def sufficient_statistics(self, x: JaxRealArray) -> PoissonEP:
        return PoissonEP(x)


@dataclass
class PoissonEP(HasConjugatePrior[PoissonNP], Samplable):
    mean: JaxRealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape

    @classmethod
    def natural_parametrization_cls(cls) -> type[PoissonNP]:
        return PoissonNP

    def to_nat(self) -> PoissonNP:
        return PoissonNP(jnp.log(self.mean))

    # The expected_carrier_measure is -exp(-mean) * sum over k from zero to infinity of
    #   lambda ** k * log(k!) / k! = lambda ** k * log Gamma(k+1) / Gamma(k+1)

    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        return jax.random.poisson(key, self.mean, shape)

    def conjugate_prior_distribution(self, n: JaxRealArray) -> GammaNP:
        return GammaNP(-n, n * self.mean)

    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.mean
