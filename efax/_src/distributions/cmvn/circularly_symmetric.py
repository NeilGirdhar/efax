from __future__ import annotations

import math

import jax.random as jr
from tjax import JaxArray, JaxComplexArray, JaxRealArray, KeyArray, Shape, outer_product
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...interfaces.multidimensional import Multidimensional
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import (SymmetricMatrixSupport, VectorSupport, complex_field,
                          distribution_parameter)


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
        SymmetricMatrixSupport(hermitian=True, negative_semidefinite=True))
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
        xp = self.array_namespace()
        log_det_s = xp.log(xp.real(xp.linalg.det(-self.negative_precision)))
        return -log_det_s + self.dimensions() * math.log(math.pi)

    @override
    def to_exp(self) -> ComplexCircularlySymmetricNormalEP:
        xp = self.array_namespace()
        return ComplexCircularlySymmetricNormalEP(
            xp.conj(xp.linalg.inv(-self.negative_precision)))

    @override
    def carrier_measure(self, x: JaxComplexArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return xp.zeros(x.shape[:-1])

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxComplexArray, **fixed_parameters: JaxArray
                              ) -> ComplexCircularlySymmetricNormalEP:
        return ComplexCircularlySymmetricNormalEP(outer_product(x, x))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxComplexArray:
        return self.to_exp().sample(key, shape)

    @override
    def dimensions(self) -> int:
        return self.negative_precision.shape[-1]


@dataclass
class ComplexCircularlySymmetricNormalEP(
        HasEntropyEP[ComplexCircularlySymmetricNormalNP],
        Multidimensional,
        Samplable):
    """The expectation parameters of the circularly symmetric complex normal distribution.

    This has zero mean and and zero pseudo-variance. This is a curved exponential family.

    Args:
        variance: Var(x).
    """
    variance: JaxComplexArray = distribution_parameter(
            SymmetricMatrixSupport(hermitian=True, positive_semidefinite=True))

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
        xp = self.array_namespace()
        return ComplexCircularlySymmetricNormalNP(xp.conj(-xp.linalg.inv(self.variance)))

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxComplexArray:
        n = self.dimensions()
        shape = self.shape if shape is None else shape + self.shape
        xy_rvs = jr.multivariate_normal(key,
                                        self._multivariate_normal_mean(),
                                        self._multivariate_normal_cov(),
                                        shape)
        return xy_rvs[..., :n] + 1j * xy_rvs[..., n:]

    @override
    def dimensions(self) -> int:
        return self.variance.shape[-1]

    def _multivariate_normal_mean(self) -> JaxRealArray:
        """Return the mean of a corresponding real distribution with double the size."""
        xp = self.array_namespace()
        n = self.dimensions()
        return xp.zeros((*self.shape, n * 2))

    def _multivariate_normal_cov(self) -> JaxRealArray:
        """Return the covariance of a corresponding real distribution with double the size."""
        xp = self.array_namespace()
        gamma_r = 0.5 * xp.real(self.variance)
        gamma_i = 0.5 * xp.imag(self.variance)
        return xp.concat([xp.concat([gamma_r, -gamma_i], axis=-1),
                          xp.concat([gamma_i, gamma_r], axis=-1)],
                         axis=-2)
