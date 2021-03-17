from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from tjax import RealArray, Shape, dataclass

from .expectation_parametrization import ExpectationParametrization
from .natural_parametrization import NaturalParametrization
from .parameter import SymmetricMatrixSupport, VectorSupport, distribution_parameter

__all__ = ['MultivariateNormalNP', 'MultivariateNormalEP']


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


@dataclass
class MultivariateNormalEP(ExpectationParametrization[MultivariateNormalNP]):
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

    # New methods ----------------------------------------------------------------------------------
    def variance(self) -> RealArray:
        return self.second_moment - _broadcasted_outer(self.mean)
