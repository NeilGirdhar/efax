from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tjax import JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...expectation_parametrization import ExpectationParametrization
from ...interfaces.multidimensional import Multidimensional
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import SymmetricMatrixSupport, VectorSupport, distribution_parameter

__all__ = ['MultivariateNormalNP', 'MultivariateNormalEP', 'MultivariateNormalVP']


def _broadcasted_outer(x: JaxRealArray) -> JaxRealArray:
    return jnp.einsum("...i,...j->...ij", x, x)


@dataclass
class MultivariateNormalNP(HasEntropyNP['MultivariateNormalEP'],
                           NaturalParametrization['MultivariateNormalEP', JaxRealArray],
                           Multidimensional):
    """The natural parametrization of the multivariate normal distribution.

    Args:
        mean_times_precision: E(x) / Var(x).
        negative_half_precision: -0.5 / Var(x).
    """
    mean_times_precision: JaxRealArray = distribution_parameter(VectorSupport())
    negative_half_precision: JaxRealArray = distribution_parameter(SymmetricMatrixSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean_times_precision.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        eta = self.mean_times_precision
        h_inv = jnp.linalg.inv(self.negative_half_precision)
        a: JaxRealArray = jnp.einsum("...i,...ij,...j", eta, h_inv, eta)
        _, ld = jnp.linalg.slogdet(-self.negative_half_precision)
        return -0.25 * a - 0.5 * ld + 0.5 * self.dimensions() * jnp.log(np.pi)

    @override
    def to_exp(self) -> MultivariateNormalEP:
        h_inv = jnp.linalg.inv(self.negative_half_precision)
        h_inv_times_eta: JaxRealArray = jnp.einsum(
            "...ij,...j->...i", h_inv, self.mean_times_precision)
        mean = -0.5 * h_inv_times_eta
        second_moment = 0.25 * _broadcasted_outer(h_inv_times_eta) - 0.5 * h_inv
        return MultivariateNormalEP(mean, second_moment)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape[:-1])

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> MultivariateNormalEP:
        return MultivariateNormalEP(x, _broadcasted_outer(x))

    @override
    def dimensions(self) -> int:
        return self.mean_times_precision.shape[-1]


@dataclass
class MultivariateNormalEP(HasEntropyEP[MultivariateNormalNP],
                           ExpectationParametrization[MultivariateNormalNP], Multidimensional,
                           Samplable):
    mean: JaxRealArray = distribution_parameter(VectorSupport())
    second_moment: JaxRealArray = distribution_parameter(SymmetricMatrixSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[MultivariateNormalNP]:
        return MultivariateNormalNP

    @override
    def to_nat(self) -> MultivariateNormalNP:
        precision = jnp.linalg.inv(self.variance())
        mean_times_precision = jnp.einsum("...ij,...j->...i", precision, self.mean)
        return MultivariateNormalNP(mean_times_precision, -0.5 * precision)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_variance_parametrization().sample(key, shape)

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    def variance(self) -> JaxRealArray:
        return self.second_moment - _broadcasted_outer(self.mean)

    def to_variance_parametrization(self) -> MultivariateNormalVP:
        return MultivariateNormalVP(self.mean, self.variance())


@dataclass
class MultivariateNormalVP(Samplable, Multidimensional):
    """The expectation parametrization of the multivariate normal distribution.

    Args:
        mean: E(x).
        variance: Var(x).
    """
    mean: JaxRealArray = distribution_parameter(VectorSupport())
    variance: JaxRealArray = distribution_parameter(SymmetricMatrixSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport()

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        return jax.random.multivariate_normal(key, self.mean, self.variance, shape)

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    def to_exp(self) -> MultivariateNormalEP:
        second_moment = self.variance + _broadcasted_outer(self.mean)
        return MultivariateNormalEP(self.mean, second_moment)
