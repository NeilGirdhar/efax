from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from tjax import RealArray, Shape
from tjax.dataclasses import dataclass

from ...conjugate_prior import HasGeneralizedConjugatePrior
from ...multidimensional import Multidimensional
from ...natural_parametrization import NaturalParametrization
from ...parameter import ScalarSupport, VectorSupport, distribution_parameter
from ...samplable import Samplable
from .diagonal import MultivariateDiagonalNormalNP
from .isotropic import IsotropicNormalNP

__all__ = ['MultivariateFixedVarianceNormalNP', 'MultivariateFixedVarianceNormalEP']


@dataclass
class MultivariateFixedVarianceNormalNP(NaturalParametrization['MultivariateFixedVarianceNormalEP',
                                                               RealArray],
                                        Multidimensional,
                                        Samplable):
    """The multivariate normal distribution with fixed variance.

    This is a curved exponential family.
    """
    mean_times_precision: RealArray = distribution_parameter(VectorSupport())
    variance: RealArray = distribution_parameter(ScalarSupport(), fixed=True)

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean_times_precision.shape[:-1]

    def log_normalizer(self) -> RealArray:
        eta = self.mean_times_precision
        return 0.5 * (jnp.sum(jnp.square(eta), axis=-1) * self.variance
                      + self.dimensions() * jnp.log(math.pi * 2.0 * self.variance))

    def to_exp(self) -> MultivariateFixedVarianceNormalEP:
        return MultivariateFixedVarianceNormalEP(
            self.mean_times_precision * self.variance[..., jnp.newaxis],
            variance=self.variance)

    def carrier_measure(self, x: RealArray) -> RealArray:
        return -0.5 * jnp.sum(jnp.square(x), axis=-1) / self.variance

    def sufficient_statistics(self, x: RealArray) -> MultivariateFixedVarianceNormalEP:
        shape = x.shape[:-1]
        variance = jnp.broadcast_to(self.variance, shape)
        return MultivariateFixedVarianceNormalEP(x, variance=variance)

    def sample(self, key: KeyArray, shape: Shape | None = None) -> RealArray:
        return self.to_exp().sample(key, shape)

    def dimensions(self) -> int:
        return self.mean_times_precision.shape[-1]


@dataclass
class MultivariateFixedVarianceNormalEP(
        HasGeneralizedConjugatePrior[MultivariateFixedVarianceNormalNP],
        Multidimensional,
        Samplable):
    mean: RealArray = distribution_parameter(VectorSupport())
    variance: RealArray = distribution_parameter(ScalarSupport(), fixed=True)

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @classmethod
    def natural_parametrization_cls(cls) -> type[MultivariateFixedVarianceNormalNP]:
        return MultivariateFixedVarianceNormalNP

    def to_nat(self) -> MultivariateFixedVarianceNormalNP:
        return MultivariateFixedVarianceNormalNP(self.mean / self.variance[..., jnp.newaxis],
                                                 variance=self.variance)

    def expected_carrier_measure(self) -> RealArray:
        return -0.5 * (jnp.sum(jnp.square(self.mean), axis=-1) / self.variance + self.dimensions())

    def sample(self, key: KeyArray, shape: Shape | None = None) -> RealArray:
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        variance = self.variance[..., jnp.newaxis]
        deviation = jnp.sqrt(variance)
        return jax.random.normal(key, shape) * deviation + self.mean

    def conjugate_prior_distribution(self, n: RealArray) -> IsotropicNormalNP:
        negative_half_precision = -0.5 * n / self.variance
        return IsotropicNormalNP(n[..., jnp.newaxis] * self.mean, negative_half_precision)

    def generalized_conjugate_prior_distribution(self, n: RealArray
                                                 ) -> MultivariateDiagonalNormalNP:
        negative_half_precision = -0.5 * n / self.variance[..., jnp.newaxis]
        return MultivariateDiagonalNormalNP(n * self.mean, negative_half_precision)

    def conjugate_prior_observation(self) -> RealArray:
        return self.mean

    def dimensions(self) -> int:
        return self.mean.shape[-1]
