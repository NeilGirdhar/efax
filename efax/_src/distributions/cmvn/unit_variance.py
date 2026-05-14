from __future__ import annotations

import math
from typing import override

import jax.random as jr
from array_api_compat import array_namespace
from tjax import JaxArray, JaxComplexArray, JaxRealArray, KeyArray, Shape, abs_square
from tjax.dataclasses import dataclass

from efax._src.analytic_continuation import analytic_abs_square
from efax._src.interfaces.multidimensional import Multidimensional
from efax._src.interfaces.samplable import Samplable
from efax._src.mixins.has_entropy import HasEntropyEP, HasEntropyNP
from efax._src.natural_parametrization import NaturalParametrization
from efax._src.parameter import VectorSupport, complex_field, distribution_parameter


@dataclass
class ComplexMultivariateUnitVarianceNormalNP(
    HasEntropyNP["ComplexMultivariateUnitVarianceNormalEP"],
    NaturalParametrization["ComplexMultivariateUnitVarianceNormalEP", JaxComplexArray],
    Multidimensional,
    Samplable,
):
    """The complex multivariate normal distribution with unit variance, and zero pseudo-variance.

    This is a curved exponential family.

    Args:
        two_mean: 2 * E(x).
    """

    two_mean: JaxComplexArray = distribution_parameter(VectorSupport(ring=complex_field))
    # S = I, U = 0
    # P = I, R = 0
    # H = -I, J = 0
    # K = 0, L = I/2
    # eta = 2mean.conjugate()
    # Leta = mean.conjugate()

    @property
    @override
    def shape(self) -> Shape:
        return self.two_mean.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport(ring=complex_field)

    @override
    def log_normalizer(self) -> JaxRealArray:
        two_mean_abs_square = analytic_abs_square(self.two_mean)
        xp = array_namespace(two_mean_abs_square)
        return 0.25 * xp.sum(two_mean_abs_square, axis=-1) + self.dimensions() * math.log(math.pi)

    @override
    def to_exp(self) -> ComplexMultivariateUnitVarianceNormalEP:
        return ComplexMultivariateUnitVarianceNormalEP(self.two_mean * 0.5)

    @override
    def carrier_measure(self, x: JaxComplexArray) -> JaxRealArray:
        xp = array_namespace(self, x)
        return -xp.sum(abs_square(x), axis=-1)

    @override
    @classmethod
    def sufficient_statistics(
        cls, x: JaxComplexArray, **fixed_parameters: JaxArray
    ) -> ComplexMultivariateUnitVarianceNormalEP:
        return ComplexMultivariateUnitVarianceNormalEP(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxComplexArray:
        return self.to_exp().sample(key, shape)

    @override
    def dimensions(self) -> int:
        return self.two_mean.shape[-1]


@dataclass
class ComplexMultivariateUnitVarianceNormalEP(
    HasEntropyEP[ComplexMultivariateUnitVarianceNormalNP], Multidimensional, Samplable
):
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
        return ComplexMultivariateUnitVarianceNormalNP(self.mean * 2.0)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = array_namespace(self)
        # The second moment of a normal distribution with the given mean.
        return -(xp.sum(abs_square(self.mean), axis=-1) + self.dimensions())

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxComplexArray:
        xp = array_namespace(self)
        shape = self.mean.shape if shape is None else shape + self.mean.shape
        grow = (xp.newaxis,) * (len(shape) - self.mean.ndim)
        key_a, key_b = jr.split(key)
        a = jr.normal(key_a, shape)
        b = jr.normal(key_b, shape)
        return a + 1j * b + self.mean[grow]

    # def conjugate_prior_distribution(self, n: JaxRealArray) -> IsotropicNormalNP:
    #     negative_half_precision = -0.5 * n * xp.ones(self.shape)
    #     return IsotropicNormalNP(n * self.mean, negative_half_precision)
    #
    # def as_conjugate_prior_observation(self) -> JaxComplexArray:
    #     return self.mean

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]
