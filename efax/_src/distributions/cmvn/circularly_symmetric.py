from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from tjax import ComplexArray, RealArray, Shape
from tjax.dataclasses import dataclass

from ...expectation_parametrization import ExpectationParametrization
from ...multidimensional import Multidimensional
from ...natural_parametrization import NaturalParametrization
from ...parameter import SymmetricMatrixSupport, distribution_parameter
from ...samplable import Samplable

__all__ = ['ComplexCircularlySymmetricNormalNP', 'ComplexCircularlySymmetricNormalEP']


def _broadcasted_outer_c(x: ComplexArray) -> ComplexArray:
    return jnp.einsum("...i,...j->...ij", x, x.conjugate())


@dataclass
class ComplexCircularlySymmetricNormalNP(
        NaturalParametrization['ComplexCircularlySymmetricNormalEP', ComplexArray],
        Multidimensional,
        Samplable):
    """The complex multivariate normal distribution with zero mean and and zero pseudo-variance.

    This is a curved exponential family.
    """
    negative_precision: ComplexArray = distribution_parameter(
        SymmetricMatrixSupport(hermitian=True))
    # S = -1/negative_precision, U = 0
    # P = S.conjugate, R = 0
    # H = -1/S, J = 0
    # K = 0, L = S/2
    # eta = 0
    # Leta = 0

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.negative_precision.shape[:-2]

    def log_normalizer(self) -> RealArray:
        log_det_s = jnp.log(jnp.linalg.det(-self.negative_precision).real)
        return -log_det_s + self.dimensions() * math.log(math.pi)

    def to_exp(self) -> ComplexCircularlySymmetricNormalEP:
        return ComplexCircularlySymmetricNormalEP(
            jnp.linalg.inv(-self.negative_precision).conjugate())

    def carrier_measure(self, x: ComplexArray) -> RealArray:
        return jnp.zeros(x.shape[:-1])

    def sufficient_statistics(self, x: ComplexArray) -> ComplexCircularlySymmetricNormalEP:
        return ComplexCircularlySymmetricNormalEP(_broadcasted_outer_c(x))

    def sample(self, key: KeyArray, shape: Shape | None = None) -> ComplexArray:
        return self.to_exp().sample(key, shape)

    def dimensions(self) -> int:
        return self.negative_precision.shape[-1]


@dataclass
class ComplexCircularlySymmetricNormalEP(
        ExpectationParametrization[ComplexCircularlySymmetricNormalNP],
        Multidimensional,
        Samplable):
    variance: ComplexArray = distribution_parameter(SymmetricMatrixSupport(hermitian=True))

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.variance.shape[:-2]

    @classmethod
    def natural_parametrization_cls(cls) -> type[ComplexCircularlySymmetricNormalNP]:
        return ComplexCircularlySymmetricNormalNP

    def to_nat(self) -> ComplexCircularlySymmetricNormalNP:
        return ComplexCircularlySymmetricNormalNP(
            -jnp.linalg.inv(self.variance).conjugate())

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape)

    def sample(self, key: KeyArray, shape: Shape | None = None) -> ComplexArray:
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

    def dimensions(self) -> int:
        return self.variance.shape[-1]

    # Private method -------------------------------------------------------------------------------
    def _multivariate_normal_mean(self) -> RealArray:
        """Return the mean of a corresponding real distribution with double the size."""
        n = self.dimensions()
        return jnp.zeros((*self.shape, n * 2))

    def _multivariate_normal_cov(self) -> RealArray:
        """Return the covariance of a corresponding real distribution with double the size."""
        gamma_r = 0.5 * self.variance.real
        gamma_i = 0.5 * self.variance.imag
        return jnp.block([[gamma_r, -gamma_i], [gamma_i, gamma_r]])
