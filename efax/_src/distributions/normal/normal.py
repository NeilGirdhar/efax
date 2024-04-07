from __future__ import annotations

from typing import Any

import jax
import numpy as np
from array_api_compat import get_namespace
from tjax import JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...expectation_parametrization import ExpectationParametrization
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import ScalarSupport, distribution_parameter, negative_support, positive_support
from ...parametrization import SimpleDistribution


@dataclass
class NormalNP(HasEntropyNP['NormalEP'],
               Samplable,
               NaturalParametrization['NormalEP', JaxRealArray],
               SimpleDistribution):
    """The natural parametrization of the normal distribution.

    Args:
        mean_times_precision: The mean times the precision.
        negative_half_precision: -0.5 times the precision.
    """
    mean_times_precision: JaxRealArray = distribution_parameter(ScalarSupport())
    negative_half_precision: JaxRealArray = distribution_parameter(ScalarSupport(
        ring=negative_support))

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
        xp = self.get_namespace()
        return (-xp.square(self.mean_times_precision) / (4.0 * self.negative_half_precision)
                + 0.5 * xp.log(-np.pi / self.negative_half_precision))

    @override
    def to_exp(self) -> NormalEP:
        xp = self.get_namespace()
        mean = -self.mean_times_precision / (2.0 * self.negative_half_precision)
        second_moment = xp.square(mean) - 0.5 / self.negative_half_precision
        return NormalEP(mean, second_moment)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.get_namespace(x)
        return xp.zeros(x.shape)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> NormalEP:
        xp = get_namespace(x)
        return NormalEP(x, xp.square(x))

    def to_var(self) -> NormalVP:
        return self.to_exp().to_var()

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_var().sample(key, shape)


@dataclass
class NormalEP(HasEntropyEP[NormalNP],
               ExpectationParametrization[NormalNP],
               Samplable,
               SimpleDistribution):
    """The expectation parametrization of the normal distribution.

    Args:
        mean: The mean.
        second_moment: The second moment.
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport())
    second_moment: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

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
        xp = self.get_namespace()
        return xp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        xp = self.get_namespace()
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        deviation = xp.sqrt(self.variance())
        return jax.random.normal(key, shape) * deviation + self.mean

    def variance(self) -> JaxRealArray:
        xp = self.get_namespace()
        return self.second_moment - xp.square(self.mean)

    def to_var(self) -> NormalVP:
        return NormalVP(self.mean, self.variance())


@dataclass
class NormalVP(Samplable, SimpleDistribution):
    """The variance parametrization of the normal distribution.

    Args:
        mean: The mean.
        variance: The variance.
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport())
    variance: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        xp = self.get_namespace()
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        deviation = xp.sqrt(self.variance)
        return jax.random.normal(key, shape) * deviation + self.mean

    def log_pdf(self, x: JaxRealArray) -> JaxRealArray:
        return self.to_nat().log_pdf(x)

    def pdf(self, x: JaxRealArray) -> JaxRealArray:
        return self.to_nat().pdf(x)

    def to_exp(self) -> NormalEP:
        xp = self.get_namespace()
        second_moment = self.variance + xp.square(self.mean)
        return NormalEP(self.mean, second_moment)

    def to_nat(self) -> NormalNP:
        precision = 1.0 / self.variance
        mean_times_precision = self.mean * precision
        negative_half_precision = -0.5 * precision
        return NormalNP(mean_times_precision, negative_half_precision)
