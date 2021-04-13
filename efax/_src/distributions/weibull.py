from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from chex import Numeric
from tjax import Generator, RealArray, Shape
from tjax.dataclasses import dataclass

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter
from ..samplable import Samplable

__all__ = ['WeibullNP', 'WeibullEP']


@dataclass
class WeibullNP(NaturalParametrization['WeibullEP']):
    concentration: Numeric = distribution_parameter(ScalarSupport(), fixed=True)
    # eta = -scale^-concentration
    eta: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.eta.shape

    def log_normalizer(self) -> RealArray:
        return -jnp.log(-self.eta) - jnp.log(self.concentration)

    def to_exp(self) -> WeibullEP:
        return WeibullEP(self.concentration, -1.0 / self.eta)

    def carrier_measure(self, x: RealArray) -> RealArray:
        return (self.concentration - 1.0) * jnp.log(x)

    def sufficient_statistics(self, x: RealArray) -> WeibullEP:
        return WeibullEP(jnp.broadcast_to(self.concentration, x.shape), x ** self.concentration)


@dataclass
class WeibullEP(ExpectationParametrization[WeibullNP], Samplable):
    concentration: Numeric = distribution_parameter(ScalarSupport(), fixed=True)
    # chi = scale^concentration
    chi: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.chi.shape

    def to_nat(self) -> WeibullNP:
        return WeibullNP(self.concentration, -1.0 / self.chi)

    def expected_carrier_measure(self) -> RealArray:
        k = self.concentration
        lambda_ = self.chi ** (1.0 / k)
        return (k - 1.0) * jnp.log(lambda_) - jnp.euler_gamma * (1.0 - 1.0 / k)

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> RealArray:
        if shape is not None:
            shape += self.shape()
        else:
            shape = self.shape()
        lambda_ = self.chi ** (1.0 / self.concentration)
        return jax.random.weibull_min(rng.key, lambda_, self.concentration, shape)
