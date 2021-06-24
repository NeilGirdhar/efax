from __future__ import annotations

from typing import Optional, Type

import jax
import jax.numpy as jnp
from jax.nn import softplus
from jax.scipy import special as jss
from tjax import Generator, RealArray, Shape
from tjax.dataclasses import dataclass

from ..exp_to_nat import ExpToNat
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter
from ..samplable import Samplable
from ..tools import inverse_softplus

__all__ = ['GammaNP', 'GammaEP']


@dataclass
class GammaNP(NaturalParametrization['GammaEP', RealArray], Samplable):
    negative_rate: RealArray = distribution_parameter(ScalarSupport())
    shape_minus_one: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.negative_rate.shape

    def log_normalizer(self) -> RealArray:
        shape = self.shape_minus_one + 1.0
        return jss.gammaln(shape) - shape * jnp.log(-self.negative_rate)

    def to_exp(self) -> GammaEP:
        shape = self.shape_minus_one + 1.0
        return GammaEP(-shape / self.negative_rate,
                       jss.digamma(shape) - jnp.log(-self.negative_rate))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape)

    def sufficient_statistics(self, x: RealArray) -> GammaEP:
        return GammaEP(x, jnp.log(x))

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> RealArray:
        if shape is not None:
            shape += self.shape
        return -jax.random.gamma(rng.key, self.shape_minus_one + 1.0, shape) / self.negative_rate


@dataclass
class GammaEP(ExpToNat[GammaNP, RealArray]):
    mean: RealArray = distribution_parameter(ScalarSupport())
    mean_log: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape

    @classmethod
    def natural_parametrization_cls(cls) -> Type[GammaNP]:
        return GammaNP

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape)

    def search_to_natural(self, search_parameters: RealArray) -> GammaNP:
        shape = softplus(search_parameters)
        rate = shape / self.mean
        return GammaNP(-rate, shape - 1.0)

    def search_gradient(self, search_parameters: RealArray) -> RealArray:
        shape = softplus(search_parameters)
        log_mean_minus_mean_log = jnp.log(self.mean) - self.mean_log
        return log_mean_minus_mean_log - jnp.log(shape) + jss.digamma(shape)
        # gradient is 1.0 / shape - jss.polygamma(1, shape)
        # where polygamma(1) is trigamma

    # Overridden methods ---------------------------------------------------------------------------
    def initial_search_parameters(self) -> RealArray:
        log_mean_minus_mean_log = jnp.log(self.mean) - self.mean_log
        initial_shape = (
            (3.0 - log_mean_minus_mean_log
             + jnp.sqrt((log_mean_minus_mean_log - 3.0) ** 2 + 24.0 * log_mean_minus_mean_log))
            / (12.0 * log_mean_minus_mean_log))
        return inverse_softplus(initial_shape)
