from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tjax import JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...expectation_parametrization import ExpectationParametrization
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import ScalarSupport, distribution_parameter

__all__ = ['NormalNP', 'NormalEP', 'NormalVP']


@dataclass
class NormalNP(HasEntropyNP['NormalEP'], NaturalParametrization['NormalEP', JaxRealArray]):
    """The natural parametrization of the normal distribution.

    Args:
        mean_times_precision: The mean times the precision.
        negative_half_precision: -0.5 times the precision.
    """
    mean_times_precision: JaxRealArray = distribution_parameter(ScalarSupport())
    negative_half_precision: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean_times_precision.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        return (-jnp.square(self.mean_times_precision) / (4.0 * self.negative_half_precision)
                + 0.5 * jnp.log(-np.pi / self.negative_half_precision))

    @override
    def to_exp(self) -> NormalEP:
        mean = -self.mean_times_precision / (2.0 * self.negative_half_precision)
        second_moment = jnp.square(mean) - 0.5 / self.negative_half_precision
        return NormalEP(mean, second_moment)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> NormalEP:
        return NormalEP(x, jnp.square(x))

    def to_var(self) -> NormalVP:
        return self.to_exp().to_var()


@dataclass
class NormalEP(HasEntropyEP[NormalNP],
               ExpectationParametrization[NormalNP],
               Samplable):
    """The expectation parametrization of the normal distribution.

    Args:
        mean: The mean.
        second_moment: The second moment.
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport())
    second_moment: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[NormalNP]:
        return NormalNP

    @override
    def to_nat(self) -> NormalNP:
        return NormalNP(self.mean / self.variance(), -0.5 / self.variance())

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        deviation = jnp.sqrt(self.variance())
        return jax.random.normal(key, shape) * deviation + self.mean

    def variance(self) -> JaxRealArray:
        return self.second_moment - jnp.square(self.mean)

    def to_var(self) -> NormalVP:
        return NormalVP(self.mean, self.variance())


@dataclass
class NormalVP(Samplable):
    """The variance parametrization of the normal distribution.

    Args:
        mean: The mean.
        variance: The variance.
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport())
    variance: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        deviation = jnp.sqrt(self.variance)
        return jax.random.normal(key, shape) * deviation + self.mean

    def log_pdf(self, x: JaxRealArray) -> JaxRealArray:
        return self.to_nat().log_pdf(x)

    def pdf(self, x: JaxRealArray) -> JaxRealArray:
        return self.to_nat().pdf(x)

    def to_exp(self) -> NormalEP:
        second_moment = self.variance + jnp.square(self.mean)
        return NormalEP(self.mean, second_moment)

    def to_nat(self) -> NormalNP:
        precision = 1.0 / self.variance
        mean_times_precision = self.mean * precision
        negative_half_precision = -0.5 * precision
        return NormalNP(mean_times_precision, negative_half_precision)
