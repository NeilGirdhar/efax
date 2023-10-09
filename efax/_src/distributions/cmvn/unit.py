from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from tjax import JaxComplexArray, JaxRealArray, KeyArray, Shape, abs_square
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...expectation_parametrization import ExpectationParametrization
from ...interfaces.multidimensional import Multidimensional
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import VectorSupport, complex_field, distribution_parameter

__all__ = ['ComplexMultivariateUnitNormalNP', 'ComplexMultivariateUnitNormalEP']


@dataclass
class ComplexMultivariateUnitNormalNP(HasEntropyNP['ComplexMultivariateUnitNormalEP'],
                                      NaturalParametrization['ComplexMultivariateUnitNormalEP',
                                                             JaxComplexArray],
                                      Multidimensional,
                                      Samplable):
    """The complex multivariate normal distribution with unit variance, and zero pseudo-variance.

    This is a curved exponential family.

    Args:
        two_mean_conjugate: 2 * E(conjugate(x)).
    """
    two_mean_conjugate: JaxComplexArray = distribution_parameter(VectorSupport(ring=complex_field))
    # S = I, U = 0
    # P = I, R = 0
    # H = -I, J = 0
    # K = 0, L = I/2
    # eta = 2mean.conjugate()
    # Leta = mean.conjugate()

    @property
    @override
    def shape(self) -> Shape:
        return self.two_mean_conjugate.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport(ring=complex_field)

    @override
    def log_normalizer(self) -> JaxRealArray:
        mean_conjugate = self.two_mean_conjugate * 0.5
        return jnp.sum(abs_square(mean_conjugate), axis=-1) + self.dimensions() * math.log(math.pi)

    @override
    def to_exp(self) -> ComplexMultivariateUnitNormalEP:
        return ComplexMultivariateUnitNormalEP(self.two_mean_conjugate.conjugate() * 0.5)

    @override
    def carrier_measure(self, x: JaxComplexArray) -> JaxRealArray:
        return -jnp.sum(abs_square(x), axis=-1)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxComplexArray, **fixed_parameters: Any
                              ) -> ComplexMultivariateUnitNormalEP:
        return ComplexMultivariateUnitNormalEP(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxComplexArray:
        return self.to_exp().sample(key, shape)

    @override
    def dimensions(self) -> int:
        return self.two_mean_conjugate.shape[-1]


@dataclass
class ComplexMultivariateUnitNormalEP(HasEntropyEP[ComplexMultivariateUnitNormalNP],
                                      ExpectationParametrization[ComplexMultivariateUnitNormalNP],
                                      Multidimensional,
                                      Samplable):
    """The complex multivariate normal distribution with unit variance, and zero pseudo-variance.

    This is a curved exponential family.

    Args:
        mean: E(x).
    """
    mean: JaxComplexArray = distribution_parameter(VectorSupport(ring=complex_field))

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport(ring=complex_field)

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[ComplexMultivariateUnitNormalNP]:
        return ComplexMultivariateUnitNormalNP

    @override
    def to_nat(self) -> ComplexMultivariateUnitNormalNP:
        return ComplexMultivariateUnitNormalNP(self.mean.conjugate() * 2.0)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        # The second moment of a normal distribution with the given mean.
        return -(jnp.sum(abs_square(self.mean), axis=-1) + self.dimensions())

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxComplexArray:
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        a = jax.random.normal(key, shape)
        b = jax.random.normal(key, shape)
        return a + 1j * b + self.mean

    # def conjugate_prior_distribution(self, n: JaxRealArray) -> IsotropicNormalNP:
    #     negative_half_precision = -0.5 * n * jnp.ones(self.shape)
    #     return IsotropicNormalNP(n * self.mean, negative_half_precision)
    #
    # def conjugate_prior_observation(self) -> JaxComplexArray:
    #     return self.mean

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]
