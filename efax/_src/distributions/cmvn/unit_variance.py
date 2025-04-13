from __future__ import annotations

import math

import jax.random as jr
from tjax import JaxArray, JaxComplexArray, JaxRealArray, KeyArray, Shape, abs_square
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...interfaces.multidimensional import Multidimensional
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import VectorSupport, complex_field, distribution_parameter


@dataclass
class ComplexMultivariateUnitVarianceNormalNP(
        HasEntropyNP['ComplexMultivariateUnitVarianceNormalEP'],
        NaturalParametrization['ComplexMultivariateUnitVarianceNormalEP', JaxComplexArray],
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
        xp = self.array_namespace()
        mean_conjugate = self.two_mean_conjugate * 0.5
        return xp.sum(abs_square(mean_conjugate), axis=-1) + self.dimensions() * math.log(math.pi)

    @override
    def to_exp(self) -> ComplexMultivariateUnitVarianceNormalEP:
        xp = self.array_namespace()
        return ComplexMultivariateUnitVarianceNormalEP(xp.conj(self.two_mean_conjugate) * 0.5)

    @override
    def carrier_measure(self, x: JaxComplexArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return -xp.sum(abs_square(x), axis=-1)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxComplexArray, **fixed_parameters: JaxArray
                              ) -> ComplexMultivariateUnitVarianceNormalEP:
        return ComplexMultivariateUnitVarianceNormalEP(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxComplexArray:
        return self.to_exp().sample(key, shape)

    @override
    def dimensions(self) -> int:
        return self.two_mean_conjugate.shape[-1]


@dataclass
class ComplexMultivariateUnitVarianceNormalEP(
        HasEntropyEP[ComplexMultivariateUnitVarianceNormalNP],
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
    def natural_parametrization_cls(cls) -> type[ComplexMultivariateUnitVarianceNormalNP]:
        return ComplexMultivariateUnitVarianceNormalNP

    @override
    def to_nat(self) -> ComplexMultivariateUnitVarianceNormalNP:
        xp = self.array_namespace()
        return ComplexMultivariateUnitVarianceNormalNP(xp.conj(self.mean) * 2.0)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.array_namespace()
        # The second moment of a normal distribution with the given mean.
        return -(xp.sum(abs_square(self.mean), axis=-1) + self.dimensions())

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxComplexArray:
        shape = self.mean.shape if shape is None else shape + self.mean.shape
        a = jr.normal(key, shape)
        b = jr.normal(key, shape)
        return a + 1j * b + self.mean

    # def conjugate_prior_distribution(self, n: JaxRealArray) -> IsotropicNormalNP:
    #     negative_half_precision = -0.5 * n * xp.ones(self.shape)
    #     return IsotropicNormalNP(n * self.mean, negative_half_precision)
    #
    # def conjugate_prior_observation(self) -> JaxComplexArray:
    #     return self.mean

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]
