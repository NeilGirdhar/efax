from __future__ import annotations

import jax.random as jr
import numpy as np
from array_api_compat import array_namespace
from tjax import JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

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
        xp = self.array_namespace()
        return (-xp.square(self.mean_times_precision) / (4.0 * self.negative_half_precision)
                + 0.5 * xp.log(-np.pi / self.negative_half_precision))

    @override
    def to_exp(self) -> NormalEP:
        xp = self.array_namespace()
        mean = -self.mean_times_precision / (2.0 * self.negative_half_precision)
        second_moment = xp.square(mean) - 0.5 / self.negative_half_precision
        return NormalEP(mean, second_moment)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return xp.zeros(x.shape)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray
                              ) -> NormalEP:
        xp = array_namespace(x)
        return NormalEP(x, xp.square(x))

    def to_variance_parametrization(self) -> NormalVP:
        return self.to_exp().to_variance_parametrization()

    def to_deviation_parametrization(self) -> NormalDP:
        return self.to_variance_parametrization().to_deviation_parametrization()

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_variance_parametrization().sample(key, shape)


@dataclass
class NormalEP(HasEntropyEP[NormalNP],
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
        xp = self.array_namespace()
        return xp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_variance_parametrization().sample(key, shape)

    def variance(self) -> JaxRealArray:
        xp = self.array_namespace()
        return self.second_moment - xp.square(self.mean)

    def to_variance_parametrization(self) -> NormalVP:
        return NormalVP(self.mean, self.variance())

    def to_deviation_parametrization(self) -> NormalDP:
        return self.to_variance_parametrization().to_deviation_parametrization()


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
        shape = self.shape if shape is None else shape + self.shape
        xp = self.array_namespace()
        deviation = xp.sqrt(self.variance)
        return jr.normal(key, shape) * deviation + self.mean

    def log_pdf(self, x: JaxRealArray) -> JaxRealArray:
        return self.to_nat().log_pdf(x)

    def pdf(self, x: JaxRealArray) -> JaxRealArray:
        return self.to_nat().pdf(x)

    def to_exp(self) -> NormalEP:
        xp = self.array_namespace()
        second_moment = self.variance + xp.square(self.mean)
        return NormalEP(self.mean, second_moment)

    def to_nat(self) -> NormalNP:
        xp = self.array_namespace()
        precision = xp.reciprocal(self.variance)
        mean_times_precision = self.mean * precision
        negative_half_precision = -0.5 * precision
        return NormalNP(mean_times_precision, negative_half_precision)

    def to_deviation_parametrization(self) -> NormalDP:
        xp = self.array_namespace()
        return NormalDP(self.mean, xp.sqrt(self.variance))


@dataclass
class NormalDP(Samplable, SimpleDistribution):
    """The deviation parametrization of the normal distribution.

    Args:
        mean: The mean.
        deviation: The standard deviation.
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport())
    deviation: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

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
        shape = self.shape if shape is None else shape + self.shape
        return jr.normal(key, shape) * self.deviation + self.mean

    def log_pdf(self, x: JaxRealArray) -> JaxRealArray:
        return self.to_nat().log_pdf(x)

    def pdf(self, x: JaxRealArray) -> JaxRealArray:
        return self.to_nat().pdf(x)

    def to_exp(self) -> NormalEP:
        return self.to_variance_parametrization().to_exp()

    def to_nat(self) -> NormalNP:
        return self.to_variance_parametrization().to_nat()

    def to_variance_parametrization(self) -> NormalVP:
        xp = self.array_namespace()
        return NormalVP(self.mean, xp.square(self.deviation))
