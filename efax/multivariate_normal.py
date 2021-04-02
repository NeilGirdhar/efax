from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from tjax import Generator, RealArray, Shape, dataclass

from .expectation_parametrization import ExpectationParametrization
from .natural_parametrization import NaturalParametrization
from .parameter import SymmetricMatrixSupport, VectorSupport, distribution_parameter
from .samplable import Samplable

__all__ = ['MultivariateNormalNP', 'MultivariateNormalEP', 'MultivariateNormalVP']


def _broadcasted_outer(x: RealArray) -> RealArray:
    return jnp.einsum("...i,...j->...ij", x, x)


@dataclass
class MultivariateNormalNP(NaturalParametrization['MultivariateNormalEP']):
    mean_times_precision: RealArray = distribution_parameter(VectorSupport())
    negative_half_precision: RealArray = distribution_parameter(SymmetricMatrixSupport())

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean_times_precision.shape[:-1]

    def log_normalizer(self) -> RealArray:
        eta = self.mean_times_precision
        k = eta.shape[-1]
        h_inv = jnp.linalg.inv(self.negative_half_precision)
        a = jnp.einsum("...i,...ij,...j", eta, h_inv, eta)
        _, ld = jnp.linalg.slogdet(-self.negative_half_precision)
        return -0.25 * a - 0.5 * ld + 0.5 * k * jnp.log(np.pi)

    def to_exp(self) -> MultivariateNormalEP:
        h_inv = jnp.linalg.inv(self.negative_half_precision)
        h_inv_times_eta = jnp.einsum("...ij,...j->...i", h_inv, self.mean_times_precision)
        mean = -0.5 * h_inv_times_eta
        second_moment = 0.25 * _broadcasted_outer(h_inv_times_eta) - 0.5 * h_inv
        return MultivariateNormalEP(mean, second_moment)

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape[:-1])

    def sufficient_statistics(self, x: RealArray) -> MultivariateNormalEP:
        return MultivariateNormalEP(x, _broadcasted_outer(x))

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.mean_times_precision.shape[-1]


@dataclass
class MultivariateNormalEP(ExpectationParametrization[MultivariateNormalNP], Samplable):
    mean: RealArray = distribution_parameter(VectorSupport())
    second_moment: RealArray = distribution_parameter(SymmetricMatrixSupport())

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    def to_nat(self) -> MultivariateNormalNP:
        precision = jnp.linalg.inv(self.variance())
        mean_times_precision = jnp.einsum("...ij,...j->...i", precision, self.mean)
        return MultivariateNormalNP(mean_times_precision, -0.5 * precision)

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape())

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> RealArray:
        return self.to_variance_parametrization().sample(rng, shape)

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    def variance(self) -> RealArray:
        return self.second_moment - _broadcasted_outer(self.mean)

    def to_variance_parametrization(self) -> MultivariateNormalVP:
        return MultivariateNormalVP(self.mean, self.variance())


@dataclass
class MultivariateNormalVP(Samplable):
    mean: RealArray = distribution_parameter(VectorSupport())
    variance: RealArray = distribution_parameter(SymmetricMatrixSupport())

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> RealArray:
        if shape is not None:
            shape += self.shape()
        else:
            shape = self.shape()
        return jax.random.multivariate_normal(rng.key, self.mean, self.variance, shape)

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    def to_exp(self) -> MultivariateNormalEP:
        second_moment = self.variance + _broadcasted_outer(self.mean)
        return MultivariateNormalEP(self.mean, second_moment)
