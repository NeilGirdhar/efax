from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from tjax import JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..conjugate_prior import HasGeneralizedConjugatePrior
from ..has_entropy import HasEntropyEP, HasEntropyNP
from ..multidimensional import Multidimensional
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, SymmetricMatrixSupport, distribution_parameter, VectorSupport
from ..samplable import Samplable

__all__ = ['WishartNP', 'WishartEP']


@dataclass
class WishartNP(HasEntropyNP,
                NaturalParametrization['WishartEP', JaxRealArray],
                Multidimensional,
                Samplable):
    """The multivariate normal distribution with fixed variance.

    This is a curved exponential family.
    """
    mean_times_precision: JaxRealArray = distribution_parameter(VectorSupport())
    variance: JaxRealArray = distribution_parameter(ScalarSupport(), fixed=True)

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean_times_precision.shape[:-1]

    @override
    def domain_support(self) -> SymmetricMatrixSupport:
        return SymmetricMatrixSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        eta = self.mean_times_precision
        return 0.5 * (jnp.sum(jnp.square(eta), axis=-1) * self.variance
                      + self.dimensions() * jnp.log(math.pi * 2.0 * self.variance))

    @override
    def to_exp(self) -> WishartEP:
        return WishartEP(
            self.mean_times_precision * self.variance[..., jnp.newaxis],
            variance=self.variance)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return -0.5 * jnp.sum(jnp.square(x), axis=-1) / self.variance

    @override
    def sufficient_statistics(self, x: JaxRealArray) -> WishartEP:
        shape = x.shape[:-1]
        variance = jnp.broadcast_to(self.variance, shape)
        return WishartEP(x, variance=variance)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)

    @override
    def dimensions(self) -> int:
        return self.mean_times_precision.shape[-1]


@dataclass
class WishartEP(
        HasEntropyEP[WishartNP],
        HasGeneralizedConjugatePrior[WishartNP],
        Multidimensional,
        Samplable):
    mean: JaxRealArray = distribution_parameter(VectorSupport())
    variance: JaxRealArray = distribution_parameter(ScalarSupport(), fixed=True)

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @override
    def domain_support(self) -> SymmetricMatrixSupport:
        return SymmetricMatrixSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[WishartNP]:
        return WishartNP

    @override
    def to_nat(self) -> WishartNP:
        return WishartNP(self.mean / self.variance[..., jnp.newaxis],
                                                 variance=self.variance)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return -0.5 * (jnp.sum(jnp.square(self.mean), axis=-1) / self.variance + self.dimensions())

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        variance = self.variance[..., jnp.newaxis]
        deviation = jnp.sqrt(variance)
        return jax.random.normal(key, shape) * deviation + self.mean

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> IsotropicNormalNP:
        negative_half_precision = -0.5 * n / self.variance
        return IsotropicNormalNP(n[..., jnp.newaxis] * self.mean, negative_half_precision)

    @override
    def generalized_conjugate_prior_distribution(self, n: JaxRealArray
                                                 ) -> MultivariateDiagonalNormalNP:
        negative_half_precision = -0.5 * n / self.variance[..., jnp.newaxis]
        return MultivariateDiagonalNormalNP(n * self.mean, negative_half_precision)

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.mean

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]

