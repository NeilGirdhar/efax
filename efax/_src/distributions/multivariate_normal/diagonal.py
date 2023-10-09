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
from ...parameter import VectorSupport, distribution_parameter

__all__ = ['MultivariateDiagonalNormalNP', 'MultivariateDiagonalNormalEP',
           'MultivariateDiagonalNormalVP']


@dataclass
class MultivariateDiagonalNormalNP(HasEntropyNP['MultivariateDiagonalNormalEP'],
                                   NaturalParametrization['MultivariateDiagonalNormalEP',
                                                          JaxRealArray],
                                   Multidimensional):
    """The natural parametrization of the normal distribution with diagonal variance.

    Args:
        mean_times_precision: E(x) / Var(x).
        negative_half_precision: The diagonal elements of -0.5 / Var(x).
    """
    mean_times_precision: JaxRealArray = distribution_parameter(VectorSupport())
    negative_half_precision: JaxRealArray = distribution_parameter(VectorSupport())

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
        components = (-jnp.square(self.mean_times_precision) / (4.0 * self.negative_half_precision)
                      + 0.5 * jnp.log(-np.pi / self.negative_half_precision))
        return jnp.sum(components, axis=-1)

    @override
    def to_exp(self) -> MultivariateDiagonalNormalEP:
        mean = -self.mean_times_precision / (2.0 * self.negative_half_precision)
        second_moment = jnp.square(mean) - 0.5 / self.negative_half_precision
        return MultivariateDiagonalNormalEP(mean, second_moment)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape[:-1])

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> MultivariateDiagonalNormalEP:
        return MultivariateDiagonalNormalEP(x, jnp.square(x))

    @override
    def dimensions(self) -> int:
        return self.mean_times_precision.shape[-1]


@dataclass
class MultivariateDiagonalNormalEP(HasEntropyEP[MultivariateDiagonalNormalNP],
                                   ExpectationParametrization[MultivariateDiagonalNormalNP],
                                   Multidimensional,
                                   Samplable):
    """The expectation parametrization of the normal distribution with diagonal variance.

    Args:
        mean: E(x).
        second moment: The diagonal elements of E(x^2).
    """
    mean: JaxRealArray = distribution_parameter(VectorSupport())
    second_moment: JaxRealArray = distribution_parameter(VectorSupport())

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
    def natural_parametrization_cls(cls) -> type[MultivariateDiagonalNormalNP]:
        return MultivariateDiagonalNormalNP

    @override
    def to_nat(self) -> MultivariateDiagonalNormalNP:
        return MultivariateDiagonalNormalNP(self.mean / self.variance(), -0.5 / self.variance())

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
        return self.second_moment - jnp.square(self.mean)

    def to_variance_parametrization(self) -> MultivariateDiagonalNormalVP:
        return MultivariateDiagonalNormalVP(self.mean, self.variance())


@dataclass
class MultivariateDiagonalNormalVP(Samplable, Multidimensional):
    """The variance parametrization of the normal distribution with diagonal variance.

    Args:
        mean: E(x).
        variance: The diagonal elements of Var(x).
    """
    mean: JaxRealArray = distribution_parameter(VectorSupport())
    variance: JaxRealArray = distribution_parameter(VectorSupport())

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
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        deviation = jnp.sqrt(self.variance)
        return jax.random.normal(key, shape) * deviation + self.mean

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    def pdf(self, x: JaxRealArray) -> JaxRealArray:
        return self.to_nat().pdf(x)

    def to_exp(self) -> MultivariateDiagonalNormalEP:
        second_moment = self.variance + jnp.square(self.mean)
        return MultivariateDiagonalNormalEP(self.mean, second_moment)

    def to_nat(self) -> MultivariateDiagonalNormalNP:
        precision = 1.0 / self.variance
        mean_times_precision = self.mean * precision
        negative_half_precision = -0.5 * precision
        return MultivariateDiagonalNormalNP(mean_times_precision, negative_half_precision)
