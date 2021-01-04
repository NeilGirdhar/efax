from __future__ import annotations

from jax import numpy as jnp
from jax.nn import softplus
from jax.scipy import special as jss
from tjax import RealArray, Shape, dataclass

from .exp_to_nat import ExpToNat
from .natural_parametrization import NaturalParametrization
from .parameter import distribution_parameter

__all__ = ['GammaNP', 'GammaEP']


@dataclass
class GammaNP(NaturalParametrization['GammaEP']):
    negative_rate: RealArray = distribution_parameter(axes=0)
    shape_minus_one: RealArray = distribution_parameter(axes=0)

    # Implemented methods --------------------------------------------------------------------------
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


@dataclass
class GammaEP(ExpToNat[GammaNP]):
    mean: RealArray = distribution_parameter(axes=0)
    mean_log: RealArray = distribution_parameter(axes=0)

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean.shape

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape())

    @classmethod
    def transform_natural_for_iteration(cls, iteration_natural: GammaNP) -> GammaNP:
        negative_rate = -softplus(-iteration_natural.negative_rate)
        shape_minus_one = softplus(iteration_natural.shape_minus_one) - 1.0
        return GammaNP(negative_rate, shape_minus_one)

    # Overridden methods ---------------------------------------------------------------------------
    def initial_natural(self) -> GammaNP:
        log_mean_minus_mean_log = jnp.log(self.mean) - self.mean_log
        initial_shape = (
            (3.0 - log_mean_minus_mean_log
             + jnp.sqrt((log_mean_minus_mean_log - 3.0) ** 2 + 24.0 * log_mean_minus_mean_log))
            / (12.0 * log_mean_minus_mean_log))
        initial_rate = initial_shape / self.mean
        return GammaNP(-initial_rate, initial_shape - 1.0)
