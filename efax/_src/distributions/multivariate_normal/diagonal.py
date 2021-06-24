from __future__ import annotations

from typing import Optional, Type

import jax
import jax.numpy as jnp
import numpy as np
from tjax import Generator, RealArray, Shape
from tjax.dataclasses import dataclass

from ...expectation_parametrization import ExpectationParametrization
from ...natural_parametrization import NaturalParametrization
from ...parameter import VectorSupport, distribution_parameter
from ...samplable import Samplable

__all__ = ['MultivariateDiagonalNormalNP', 'MultivariateDiagonalNormalEP',
           'MultivariateDiagonalNormalVP']


@dataclass
class MultivariateDiagonalNormalNP(NaturalParametrization['MultivariateDiagonalNormalEP',
                                                          RealArray]):
    mean_times_precision: RealArray = distribution_parameter(VectorSupport())
    negative_half_precision: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean_times_precision.shape[:-1]

    def log_normalizer(self) -> RealArray:
        components = (-jnp.square(self.mean_times_precision) / (4.0 * self.negative_half_precision)
                      + 0.5 * jnp.log(-np.pi / self.negative_half_precision))
        return jnp.sum(components, axis=-1)

    def to_exp(self) -> MultivariateDiagonalNormalEP:
        mean = -self.mean_times_precision / (2.0 * self.negative_half_precision)
        second_moment = jnp.square(mean) - 0.5 / self.negative_half_precision
        return MultivariateDiagonalNormalEP(mean, second_moment)

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape[:-1])

    def sufficient_statistics(self, x: RealArray) -> MultivariateDiagonalNormalEP:
        return MultivariateDiagonalNormalEP(x, jnp.square(x))

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.mean_times_precision.shape[-1]


@dataclass
class MultivariateDiagonalNormalEP(ExpectationParametrization[MultivariateDiagonalNormalNP],
                                   Samplable):
    mean: RealArray = distribution_parameter(VectorSupport())
    second_moment: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @classmethod
    def natural_parametrization_cls(cls) -> Type[MultivariateDiagonalNormalNP]:
        return MultivariateDiagonalNormalNP

    def to_nat(self) -> MultivariateDiagonalNormalNP:
        return MultivariateDiagonalNormalNP(self.mean / self.variance(), -0.5 / self.variance())

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape)

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> RealArray:
        return self.to_variance_parametrization().sample(rng, shape)

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    def variance(self) -> RealArray:
        return self.second_moment - jnp.square(self.mean)

    def to_variance_parametrization(self) -> MultivariateDiagonalNormalVP:
        return MultivariateDiagonalNormalVP(self.mean, self.variance())


@dataclass
class MultivariateDiagonalNormalVP(Samplable):
    mean: RealArray = distribution_parameter(VectorSupport())
    variance: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> RealArray:
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        deviation = jnp.sqrt(self.variance)
        return jax.random.normal(rng.key, shape) * deviation + self.mean

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    def to_exp(self) -> MultivariateDiagonalNormalEP:
        second_moment = self.variance + jnp.square(self.mean)
        return MultivariateDiagonalNormalEP(self.mean, second_moment)
