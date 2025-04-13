from __future__ import annotations

import math

import jax.random as jr
from tjax import JaxArray, JaxComplexArray, JaxRealArray, KeyArray, Shape, abs_square
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import ScalarSupport, complex_field, distribution_parameter


@dataclass
class ComplexUnitVarianceNormalNP(
        HasEntropyNP['ComplexUnitVarianceNormalEP'],
        NaturalParametrization['ComplexUnitVarianceNormalEP', JaxComplexArray],
        Samplable):
    """The natural parametrization of the complex unit normal distribution.

    It has unit variance, and zero pseudo-variance. This is a curved exponential family.

    Args:
        two_mean_conjugate: 2 * E(conjugate(x)).
    """
    two_mean_conjugate: JaxComplexArray = distribution_parameter(ScalarSupport(ring=complex_field))
    # S = I, U = 0
    # P = I, R = 0
    # H = -I, J = 0
    # K = 0, L = I/2
    # eta = 2mean.conjugate()
    # Leta = mean.conjugate()

    @property
    @override
    def shape(self) -> Shape:
        return self.two_mean_conjugate.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=complex_field)

    @override
    def log_normalizer(self) -> JaxRealArray:
        mean_conjugate = self.two_mean_conjugate * 0.5
        return abs_square(mean_conjugate) + math.log(math.pi)

    @override
    def to_exp(self) -> ComplexUnitVarianceNormalEP:
        xp = self.array_namespace()
        return ComplexUnitVarianceNormalEP(xp.conj(self.two_mean_conjugate) * 0.5)

    @override
    def carrier_measure(self, x: JaxComplexArray) -> JaxRealArray:
        return -abs_square(x)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxComplexArray, **fixed_parameters: JaxArray
                              ) -> ComplexUnitVarianceNormalEP:
        return ComplexUnitVarianceNormalEP(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxComplexArray:
        return self.to_exp().sample(key, shape)


@dataclass
class ComplexUnitVarianceNormalEP(
        HasEntropyEP[ComplexUnitVarianceNormalNP],
        Samplable):
    """The expectation parametrization of the complex unit normal distribution.

    It has unit variance, and zero pseudo-variance. This is a curved exponential family.

    Args:
        mean: E(x).
    """
    mean: JaxComplexArray = distribution_parameter(ScalarSupport(ring=complex_field))

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=complex_field)

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[ComplexUnitVarianceNormalNP]:
        return ComplexUnitVarianceNormalNP

    @override
    def to_nat(self) -> ComplexUnitVarianceNormalNP:
        xp = self.array_namespace()
        return ComplexUnitVarianceNormalNP(xp.conj(self.mean) * 2.0)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        # The second moment of a normal distribution with the given mean.
        return -(abs_square(self.mean) + 1.0)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxComplexArray:
        shape = self.shape if shape is None else shape + self.shape
        a = jr.normal(key, shape)
        b = jr.normal(key, shape)
        return a + 1j * b + self.mean

    # def conjugate_prior_distribution(self, n: JaxRealArray) -> IsotropicNormalNP:
    #     negative_half_precision = -0.5 * n * xp.ones(self.shape)
    #     return IsotropicNormalNP(n * self.mean, negative_half_precision)
    #
    # def conjugate_prior_observation(self) -> JaxComplexArray:
    #     return self.mean
