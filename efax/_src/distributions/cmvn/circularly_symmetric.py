from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from tjax import JaxComplexArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...expectation_parametrization import ExpectationParametrization
from ...interfaces.multidimensional import Multidimensional
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import (SymmetricMatrixSupport, VectorSupport, complex_field,
                          distribution_parameter)

__all__ = ['ComplexCircularlySymmetricNormalNP', 'ComplexCircularlySymmetricNormalEP']


def _broadcasted_outer_c(x: JaxComplexArray) -> JaxComplexArray:
    return jnp.einsum("...i,...j->...ij", x, x.conjugate())


@dataclass
class ComplexCircularlySymmetricNormalNP(
        HasEntropyNP['ComplexCircularlySymmetricNormalEP'],
        NaturalParametrization['ComplexCircularlySymmetricNormalEP', JaxComplexArray],
        Multidimensional,
        Samplable):
    """The natural parameters of the circularly symmetric complex multivariate normal distribution.

    This has zero mean and and zero pseudo-variance. This is a curved exponential family.

    Args:
        negative_precision: -1/Var(x).
    """
    negative_precision: JaxComplexArray = distribution_parameter(
        SymmetricMatrixSupport(hermitian=True))
    # S = -1/negative_precision, U = 0
    # P = S.conjugate, R = 0
    # H = -1/S, J = 0
    # K = 0, L = S/2
    # eta = 0
    # Leta = 0

    @property
    @override
    def shape(self) -> Shape:
        return self.negative_precision.shape[:-2]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport(ring=complex_field)

    @override
    def log_normalizer(self) -> JaxRealArray:
        log_det_s = jnp.log(jnp.linalg.det(-self.negative_precision).real)
        return -log_det_s + self.dimensions() * math.log(math.pi)

    @override
    def to_exp(self) -> ComplexCircularlySymmetricNormalEP:
        return ComplexCircularlySymmetricNormalEP(
            jnp.linalg.inv(-self.negative_precision).conjugate())

    @override
    def carrier_measure(self, x: JaxComplexArray) -> JaxRealArray:
        return jnp.zeros(x.shape[:-1])

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxComplexArray, **fixed_parameters: Any
                              ) -> ComplexCircularlySymmetricNormalEP:
        return ComplexCircularlySymmetricNormalEP(_broadcasted_outer_c(x))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxComplexArray:
        return self.to_exp().sample(key, shape)

    @override
    def dimensions(self) -> int:
        return self.negative_precision.shape[-1]


@dataclass
class ComplexCircularlySymmetricNormalEP(
        HasEntropyEP[ComplexCircularlySymmetricNormalNP],
        ExpectationParametrization[ComplexCircularlySymmetricNormalNP],
        Multidimensional,
        Samplable):
    """The expectation parameters of the circularly symmetric complex normal distribution.

    This has zero mean and and zero pseudo-variance. This is a curved exponential family.

    Args:
        variance: Var(x).
    """
    variance: JaxComplexArray = distribution_parameter(SymmetricMatrixSupport(hermitian=True))

    @property
    @override
    def shape(self) -> Shape:
        return self.variance.shape[:-2]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport(ring=complex_field)

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[ComplexCircularlySymmetricNormalNP]:
        return ComplexCircularlySymmetricNormalNP

    @override
    def to_nat(self) -> ComplexCircularlySymmetricNormalNP:
        return ComplexCircularlySymmetricNormalNP(
            -jnp.linalg.inv(self.variance).conjugate())

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxComplexArray:
        n = self.dimensions()
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        xy_rvs = jax.random.multivariate_normal(key,
                                                self._multivariate_normal_mean(),
                                                self._multivariate_normal_cov(),
                                                shape)
        return xy_rvs[..., :n] + 1j * xy_rvs[..., n:]

    @override
    def dimensions(self) -> int:
        return self.variance.shape[-1]

    def _multivariate_normal_mean(self) -> JaxRealArray:
        """Return the mean of a corresponding real distribution with double the size."""
        n = self.dimensions()
        return jnp.zeros((*self.shape, n * 2))

    def _multivariate_normal_cov(self) -> JaxRealArray:
        """Return the covariance of a corresponding real distribution with double the size."""
        gamma_r = 0.5 * self.variance.real
        gamma_i = 0.5 * self.variance.imag
        return jnp.block([[gamma_r, -gamma_i], [gamma_i, gamma_r]])
