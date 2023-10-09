from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from tjax import JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...expectation_parametrization import ExpectationParametrization
from ...interfaces.multidimensional import Multidimensional
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import ScalarSupport, VectorSupport, distribution_parameter

__all__ = ['IsotropicNormalNP', 'IsotropicNormalEP']


@dataclass
class IsotropicNormalNP(HasEntropyNP['IsotropicNormalEP'],
                        NaturalParametrization['IsotropicNormalEP', JaxRealArray],
                        Multidimensional):
    """The natural parametrization of the multivariate normal distribution with Var(x) = kI.

    Args:
        mean_times_precision: E(x) / Var(x).
        negative_half_precision: -0.5 / Var(x).
    """
    mean_times_precision: JaxRealArray = distribution_parameter(VectorSupport())
    negative_half_precision: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.negative_half_precision.shape

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        eta = self.mean_times_precision
        return 0.5 * (-0.5 * jnp.sum(jnp.square(eta), axis=-1) / self.negative_half_precision
                      + self.dimensions() * jnp.log(jnp.pi / -self.negative_half_precision))

    @override
    def to_exp(self) -> IsotropicNormalEP:
        precision = -2.0 * self.negative_half_precision
        mean = self.mean_times_precision / precision[..., jnp.newaxis]
        total_variance = self.dimensions() / precision
        total_second_moment = jnp.sum(jnp.square(mean), axis=-1) + total_variance
        return IsotropicNormalEP(mean, total_second_moment)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape[:-1])

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> IsotropicNormalEP:
        return IsotropicNormalEP(x, jnp.sum(jnp.square(x), axis=-1))

    @override
    def dimensions(self) -> int:
        return self.mean_times_precision.shape[-1]


@dataclass
class IsotropicNormalEP(HasEntropyEP[IsotropicNormalNP],
                        ExpectationParametrization[IsotropicNormalNP],
                        Samplable,
                        Multidimensional):
    """The expectation parametrization of the multivariate normal distribution with Var(x) = kI.

    Args:
        mean: E(x).
        variance: Var(x).
    """
    mean: JaxRealArray = distribution_parameter(VectorSupport())
    total_second_moment: JaxRealArray = distribution_parameter(ScalarSupport())

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
    def natural_parametrization_cls(cls) -> type[IsotropicNormalNP]:
        return IsotropicNormalNP

    @override
    def to_nat(self) -> IsotropicNormalNP:
        variance = self.variance()
        negative_half_precision = -0.5 / variance
        mean_times_precision = self.mean / variance[..., jnp.newaxis]
        return IsotropicNormalNP(mean_times_precision, negative_half_precision)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        deviation = jnp.sqrt(self.variance())
        return jax.random.normal(key, shape) * deviation + self.mean

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    def variance(self) -> JaxRealArray:
        dimensions = self.dimensions()
        return (self.total_second_moment - jnp.sum(jnp.square(self.mean), axis=-1)) / dimensions
