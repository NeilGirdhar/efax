from __future__ import annotations

import jax.random as jr
from array_api_compat import array_namespace
from tjax import JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...interfaces.multidimensional import Multidimensional
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import (ScalarSupport, VectorSupport, distribution_parameter, negative_support,
                          positive_support)


@dataclass
class IsotropicNormalNP(HasEntropyNP['IsotropicNormalEP'],
                        Samplable,
                        NaturalParametrization['IsotropicNormalEP', JaxRealArray],
                        Multidimensional):
    """The natural parametrization of the multivariate normal distribution with Var(x) = kI.

    Args:
        mean_times_precision: E(x) / Var(x).
        negative_half_precision: -0.5 / Var(x).
    """
    mean_times_precision: JaxRealArray = distribution_parameter(VectorSupport())
    negative_half_precision: JaxRealArray = distribution_parameter(ScalarSupport(
        ring=negative_support))

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
        xp = self.array_namespace()
        eta = self.mean_times_precision
        return 0.5 * (-0.5 * xp.sum(xp.square(eta), axis=-1) / self.negative_half_precision
                      + self.dimensions() * xp.log(xp.pi / -self.negative_half_precision))

    @override
    def to_exp(self) -> IsotropicNormalEP:
        xp = self.array_namespace()
        precision = -2.0 * self.negative_half_precision
        mean = self.mean_times_precision / precision[..., xp.newaxis]
        total_variance = self.dimensions() / precision
        total_second_moment = xp.sum(xp.square(mean), axis=-1) + total_variance
        return IsotropicNormalEP(mean, total_second_moment)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = array_namespace(x)
        return xp.zeros(x.shape[:-1])

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray
                              ) -> IsotropicNormalEP:
        xp = array_namespace(x)
        return IsotropicNormalEP(x, xp.sum(xp.square(x), axis=-1))

    @override
    def dimensions(self) -> int:
        return self.mean_times_precision.shape[-1]

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)


@dataclass
class IsotropicNormalEP(HasEntropyEP[IsotropicNormalNP],
                        Samplable,
                        Multidimensional):
    """The expectation parametrization of the multivariate normal distribution with Var(x) = kI.

    Args:
        mean: E(x).
        variance: Var(x).
    """
    mean: JaxRealArray = distribution_parameter(VectorSupport())
    total_second_moment: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

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
        xp = self.array_namespace()
        variance = self.variance()
        negative_half_precision = -0.5 / variance
        mean_times_precision = self.mean / variance[..., xp.newaxis]
        return IsotropicNormalNP(mean_times_precision, negative_half_precision)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        shape = self.mean.shape if shape is None else shape + self.mean.shape
        xp = self.array_namespace()
        deviation = xp.sqrt(self.variance())
        return jr.normal(key, shape) * deviation[..., xp.newaxis] + self.mean

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    def variance(self) -> JaxRealArray:
        xp = self.array_namespace()
        dimensions = self.dimensions()
        return (self.total_second_moment - xp.sum(xp.square(self.mean), axis=-1)) / dimensions
