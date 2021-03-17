from __future__ import annotations

import jax.numpy as jnp
from tjax import RealArray, Shape, dataclass

from .expectation_parametrization import ExpectationParametrization
from .natural_parametrization import NaturalParametrization
from .parameter import ScalarSupport, VectorSupport, distribution_parameter

__all__ = ['IsotropicNormalNP', 'IsotropicNormalEP']


@dataclass
class IsotropicNormalNP(NaturalParametrization['IsotropicNormalEP']):
    mean_times_precision: RealArray = distribution_parameter(VectorSupport())
    negative_half_precision: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.negative_half_precision.shape

    def log_normalizer(self) -> RealArray:
        eta = self.mean_times_precision
        num_parameters = eta.shape[-1]
        return 0.5 * (-0.5 * jnp.sum(jnp.square(eta), axis=-1) / self.negative_half_precision
                      + num_parameters * jnp.log(jnp.pi / -self.negative_half_precision))

    def to_exp(self) -> IsotropicNormalEP:
        num_parameters = self.mean_times_precision.shape[-1]
        precision = -2.0 * self.negative_half_precision
        mean = self.mean_times_precision / precision
        total_variance = num_parameters / precision
        total_second_moment = jnp.sum(jnp.square(mean), axis=-1) + total_variance
        return IsotropicNormalEP(mean, total_second_moment)

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape[:-1])

    def sufficient_statistics(self, x: RealArray) -> IsotropicNormalEP:
        return IsotropicNormalEP(x, jnp.sum(jnp.square(x), axis=-1))


@dataclass
class IsotropicNormalEP(ExpectationParametrization[IsotropicNormalNP]):
    mean: RealArray = distribution_parameter(VectorSupport())
    total_second_moment: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    def to_nat(self) -> IsotropicNormalNP:
        variance = self.variance()
        negative_half_precision = -0.5 / variance
        mean_times_precision = self.mean / variance
        return IsotropicNormalNP(mean_times_precision, negative_half_precision)

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape())

    # New methods ----------------------------------------------------------------------------------
    def variance(self) -> RealArray:
        num_parameters = self.mean.shape[-1]
        return (self.total_second_moment - jnp.sum(jnp.square(self.mean), axis=-1)) / num_parameters
