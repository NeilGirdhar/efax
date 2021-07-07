from __future__ import annotations

from typing import Optional, Type

import jax
import jax.numpy as jnp
from tjax import Generator, RealArray, Shape
from tjax.dataclasses import dataclass

from ...expectation_parametrization import ExpectationParametrization
from ...natural_parametrization import NaturalParametrization
from ...parameter import ScalarSupport, VectorSupport, distribution_parameter
from ...samplable import Samplable

__all__ = ['IsotropicNormalNP', 'IsotropicNormalEP']


@dataclass
class IsotropicNormalNP(NaturalParametrization['IsotropicNormalEP', RealArray]):
    mean_times_precision: RealArray = distribution_parameter(VectorSupport())
    negative_half_precision: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
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
        mean = self.mean_times_precision / precision[..., jnp.newaxis]
        total_variance = num_parameters / precision
        total_second_moment = jnp.sum(jnp.square(mean), axis=-1) + total_variance
        return IsotropicNormalEP(mean, total_second_moment)

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape[:-1])

    def sufficient_statistics(self, x: RealArray) -> IsotropicNormalEP:
        return IsotropicNormalEP(x, jnp.sum(jnp.square(x), axis=-1))

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.mean_times_precision.shape[-1]


@dataclass
class IsotropicNormalEP(ExpectationParametrization[IsotropicNormalNP], Samplable):
    mean: RealArray = distribution_parameter(VectorSupport())
    total_second_moment: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @classmethod
    def natural_parametrization_cls(cls) -> Type[IsotropicNormalNP]:
        return IsotropicNormalNP

    def to_nat(self) -> IsotropicNormalNP:
        variance = self.variance()
        negative_half_precision = -0.5 / variance
        mean_times_precision = self.mean / variance[..., jnp.newaxis]
        return IsotropicNormalNP(mean_times_precision, negative_half_precision)

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape)

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> RealArray:
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        deviation = jnp.sqrt(self.variance())
        return jax.random.normal(rng.key, shape) * deviation + self.mean

    # New methods ----------------------------------------------------------------------------------
    def variance(self) -> RealArray:
        num_parameters = self.mean.shape[-1]
        return (self.total_second_moment - jnp.sum(jnp.square(self.mean), axis=-1)) / num_parameters

    def dimensions(self) -> int:
        return self.mean.shape[-1]
