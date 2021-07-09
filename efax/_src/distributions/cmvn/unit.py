from __future__ import annotations

import math
from typing import Optional, Type

import jax
import jax.numpy as jnp
from tjax import ComplexArray, Generator, RealArray, Shape, abs_square
from tjax.dataclasses import dataclass

from ...expectation_parametrization import ExpectationParametrization
from ...natural_parametrization import NaturalParametrization
from ...parameter import VectorSupport, distribution_parameter
from ...samplable import Samplable

__all__ = ['ComplexMultivariateUnitNormalNP', 'ComplexMultivariateUnitNormalEP']


@dataclass
class ComplexMultivariateUnitNormalNP(NaturalParametrization['ComplexMultivariateUnitNormalEP',
                                                             ComplexArray],
                                      Samplable):
    """
    The complex multivariate normal distribution with unit variance, and zero pseudo-variance.  This
    is a curved exponential family.
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
        num_parameters = self.two_mean_conjugate.shape[-1]
        mean_conjugate = self.two_mean_conjugate * 0.5
        return jnp.sum(abs_square(mean_conjugate), axis=-1) + num_parameters * math.log(math.pi)

    def to_exp(self) -> ComplexMultivariateUnitNormalEP:
        return ComplexMultivariateUnitNormalEP(self.two_mean_conjugate.conjugate() * 0.5)

    def carrier_measure(self, x: ComplexArray) -> RealArray:
        return -jnp.sum(abs_square(x), axis=-1)

    def sufficient_statistics(self, x: ComplexArray) -> ComplexMultivariateUnitNormalEP:
        return ComplexMultivariateUnitNormalEP(x)

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> ComplexArray:
        return self.to_exp().sample(rng, shape)

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.two_mean_conjugate.shape[-1]


@dataclass
class ComplexMultivariateUnitNormalEP(ExpectationParametrization[ComplexMultivariateUnitNormalNP],
                                      Samplable):
    mean: ComplexArray = distribution_parameter(VectorSupport(is_complex=True))

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @classmethod
    def natural_parametrization_cls(cls) -> Type[ComplexMultivariateUnitNormalNP]:
        return ComplexMultivariateUnitNormalNP

    def to_nat(self) -> ComplexMultivariateUnitNormalNP:
        return ComplexMultivariateUnitNormalNP(self.mean.conjugate() * 2.0)

    def expected_carrier_measure(self) -> RealArray:
        num_parameters = self.mean.shape[-1]
        # The second moment of a normal distribution with the given mean.
        return -(jnp.sum(abs_square(self.mean), axis=-1) + num_parameters)

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> ComplexArray:
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        a = jax.random.normal(rng.key, shape)
        b = jax.random.normal(rng.key, shape)
        return a + 1j * b + self.mean

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    # Overridden methods ---------------------------------------------------------------------------
    # def conjugate_prior_distribution(self, n: RealArray) -> IsotropicNormalNP:
    #     negative_half_precision = -0.5 * n * jnp.ones(self.shape)
    #     return IsotropicNormalNP(n * self.mean, negative_half_precision)
    #
    # def conjugate_prior_observation(self) -> ComplexArray:
    #     return self.mean
