from __future__ import annotations

import math
from typing import Optional, Type

import jax
import jax.numpy as jnp
from tjax import ComplexArray, Generator, RealArray, Shape
from tjax.dataclasses import dataclass

from ...expectation_parametrization import ExpectationParametrization
from ...natural_parametrization import NaturalParametrization
from ...parameter import SymmetricMatrixSupport, distribution_parameter
from ...samplable import Samplable

__all__ = ['ComplexCircularlySymmetricNormalNP', 'ComplexCircularlySymmetricNormalEP']


def _broadcasted_outer_c(x: ComplexArray) -> ComplexArray:
    return jnp.einsum("...i,...j->...ij", x, x.conjugate())


@dataclass
class ComplexCircularlySymmetricNormalNP(
        NaturalParametrization['ComplexCircularlySymmetricNormalEP', ComplexArray],
        Samplable):
    """
    The complex multivariate normal distribution with zero mean and and zero pseudo-variance.  This
    is a curved exponential family.
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
        num_parameters = self.negative_precision.shape[-1]
        log_det_s = jnp.log(jnp.linalg.det(-self.negative_precision).real)
        return -log_det_s + num_parameters * math.log(math.pi)

    def to_exp(self) -> ComplexCircularlySymmetricNormalEP:
        return ComplexCircularlySymmetricNormalEP(
            jnp.linalg.inv(-self.negative_precision).conjugate())

    def carrier_measure(self, x: ComplexArray) -> RealArray:
        return jnp.zeros(x.shape[:-1])

    def sufficient_statistics(self, x: ComplexArray) -> ComplexCircularlySymmetricNormalEP:
        return ComplexCircularlySymmetricNormalEP(_broadcasted_outer_c(x))

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> ComplexArray:
        return self.to_exp().sample(rng, shape)

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.negative_precision.shape[-1]


@dataclass
class ComplexCircularlySymmetricNormalEP(
        ExpectationParametrization[ComplexCircularlySymmetricNormalNP], Samplable):
    variance: ComplexArray = distribution_parameter(SymmetricMatrixSupport(hermitian=True))

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.variance.shape[:-2]

    @classmethod
    def natural_parametrization_cls(cls) -> Type[ComplexCircularlySymmetricNormalNP]:
        return ComplexCircularlySymmetricNormalNP

    def to_nat(self) -> ComplexCircularlySymmetricNormalNP:
        return ComplexCircularlySymmetricNormalNP(
            -jnp.linalg.inv(self.variance).conjugate())

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape)

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> ComplexArray:
        if shape is not None:
            shape += self.variance.shape[:-1]
        else:
            shape = self.variance.shape[:-1]
        mean = jnp.zeros(shape)
        return jax.random.multivariate_normal(rng.key, mean, self.variance,
                                              shape[:-1])  # type: ignore[return-value]

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.variance.shape[-1]
