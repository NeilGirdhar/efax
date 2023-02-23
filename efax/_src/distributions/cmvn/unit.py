from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from tjax import ComplexArray, RealArray, Shape, abs_square
from tjax.dataclasses import dataclass

from ...expectation_parametrization import ExpectationParametrization
from ...multidimensional import Multidimensional
from ...natural_parametrization import NaturalParametrization
from ...parameter import VectorSupport, distribution_parameter
from ...samplable import Samplable

__all__ = ['ComplexMultivariateUnitNormalNP', 'ComplexMultivariateUnitNormalEP']


@dataclass
class ComplexMultivariateUnitNormalNP(NaturalParametrization['ComplexMultivariateUnitNormalEP',
                                                             ComplexArray],
                                      Multidimensional,
                                      Samplable):
    """The complex multivariate normal distribution with unit variance, and zero pseudo-variance.

    This is a curved exponential family.
    """
    two_mean_conjugate: ComplexArray = distribution_parameter(VectorSupport(is_complex=True))
    # S = I, U = 0
    # P = I, R = 0
    # H = -I, J = 0
    # K = 0, L = I/2
    # eta = 2mean.conjugate()
    # Leta = mean.conjugate()

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.two_mean_conjugate.shape[:-1]

    def log_normalizer(self) -> RealArray:
        mean_conjugate = self.two_mean_conjugate * 0.5
        return jnp.sum(abs_square(mean_conjugate), axis=-1) + self.dimensions() * math.log(math.pi)

    def to_exp(self) -> ComplexMultivariateUnitNormalEP:
        return ComplexMultivariateUnitNormalEP(self.two_mean_conjugate.conjugate() * 0.5)

    def carrier_measure(self, x: ComplexArray) -> RealArray:
        return -jnp.sum(abs_square(x), axis=-1)

    def sufficient_statistics(self, x: ComplexArray) -> ComplexMultivariateUnitNormalEP:
        return ComplexMultivariateUnitNormalEP(x)

    def sample(self, key: KeyArray, shape: Shape | None = None) -> ComplexArray:
        return self.to_exp().sample(key, shape)

    def dimensions(self) -> int:
        return self.two_mean_conjugate.shape[-1]


@dataclass
class ComplexMultivariateUnitNormalEP(ExpectationParametrization[ComplexMultivariateUnitNormalNP],
                                      Multidimensional,
                                      Samplable):
    mean: ComplexArray = distribution_parameter(VectorSupport(is_complex=True))

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @classmethod
    def natural_parametrization_cls(cls) -> type[ComplexMultivariateUnitNormalNP]:
        return ComplexMultivariateUnitNormalNP

    def to_nat(self) -> ComplexMultivariateUnitNormalNP:
        return ComplexMultivariateUnitNormalNP(self.mean.conjugate() * 2.0)

    def expected_carrier_measure(self) -> RealArray:
        # The second moment of a normal distribution with the given mean.
        return -(jnp.sum(abs_square(self.mean), axis=-1) + self.dimensions())

    def sample(self, key: KeyArray, shape: Shape | None = None) -> ComplexArray:
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        a = jax.random.normal(key, shape)
        b = jax.random.normal(key, shape)
        return a + 1j * b + self.mean

    # def conjugate_prior_distribution(self, n: RealArray) -> IsotropicNormalNP:
    #     negative_half_precision = -0.5 * n * jnp.ones(self.shape)
    #     return IsotropicNormalNP(n * self.mean, negative_half_precision)
    #
    # def conjugate_prior_observation(self) -> ComplexArray:
    #     return self.mean

    def dimensions(self) -> int:
        return self.mean.shape[-1]
