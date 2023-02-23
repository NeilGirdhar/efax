from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import KeyArray
from tjax import RealArray, Shape
from tjax.dataclasses import dataclass

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter
from ..samplable import Samplable

__all__ = ['NormalNP', 'NormalEP']


@dataclass
class NormalNP(NaturalParametrization['NormalEP', RealArray]):
    mean_times_precision: RealArray = distribution_parameter(ScalarSupport())
    negative_half_precision: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean_times_precision.shape

    def log_normalizer(self) -> RealArray:
        return (-jnp.square(self.mean_times_precision) / (4.0 * self.negative_half_precision)
                + 0.5 * jnp.log(-np.pi / self.negative_half_precision))

    def to_exp(self) -> NormalEP:
        mean = -self.mean_times_precision / (2.0 * self.negative_half_precision)
        second_moment = jnp.square(mean) - 0.5 / self.negative_half_precision
        return NormalEP(mean, second_moment)

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape)

    def sufficient_statistics(self, x: RealArray) -> NormalEP:
        return NormalEP(x, jnp.square(x))


@dataclass
class NormalEP(ExpectationParametrization[NormalNP], Samplable):
    mean: RealArray = distribution_parameter(ScalarSupport())
    second_moment: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape

    @classmethod
    def natural_parametrization_cls(cls) -> type[NormalNP]:
        return NormalNP

    def to_nat(self) -> NormalNP:
        return NormalNP(self.mean / self.variance(), -0.5 / self.variance())

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape)

    def sample(self, key: KeyArray, shape: Shape | None = None) -> RealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        deviation = jnp.sqrt(self.variance())
        return jax.random.normal(key, shape) * deviation + self.mean

    # New methods ----------------------------------------------------------------------------------
    def variance(self) -> RealArray:
        return self.second_moment - jnp.square(self.mean)
