from __future__ import annotations

import jax.random as jr
import numpy as np
from array_api_compat import array_namespace
from tjax import JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...interfaces.multidimensional import Multidimensional
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import VectorSupport, distribution_parameter, negative_support, positive_support


@dataclass
class MultivariateDiagonalNormalNP(HasEntropyNP['MultivariateDiagonalNormalEP'],
                                   NaturalParametrization['MultivariateDiagonalNormalEP',
                                                          JaxRealArray],
                                   Multidimensional,
                                   Samplable):
    """The natural parametrization of the normal distribution with diagonal variance.

    This distribution exists because it's the generalized conjugate prior of
    MultivariateFixedVarianceNormal.

    Args:
        mean_times_precision: E(x) / Var(x).
        negative_half_precision: The diagonal elements of -0.5 / Var(x).
    """
    mean_times_precision: JaxRealArray = distribution_parameter(VectorSupport())
    negative_half_precision: JaxRealArray = distribution_parameter(VectorSupport(
        ring=negative_support))

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
        xp = self.array_namespace()
        components = (-xp.square(self.mean_times_precision) / (4.0 * self.negative_half_precision)
                      + 0.5 * xp.log(-np.pi / self.negative_half_precision))
        return xp.sum(components, axis=-1)

    @override
    def to_exp(self) -> MultivariateDiagonalNormalEP:
        xp = self.array_namespace()
        mean = -self.mean_times_precision / (2.0 * self.negative_half_precision)
        second_moment = xp.square(mean) - 0.5 / self.negative_half_precision
        return MultivariateDiagonalNormalEP(mean, second_moment)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return xp.zeros(x.shape[:-1])

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray
                              ) -> MultivariateDiagonalNormalEP:
        xp = array_namespace(x)
        return MultivariateDiagonalNormalEP(x, xp.square(x))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)

    @override
    def dimensions(self) -> int:
        return self.mean_times_precision.shape[-1]

    def to_variance_parametrization(self) -> MultivariateDiagonalNormalVP:
        return self.to_exp().to_variance_parametrization()


@dataclass
class MultivariateDiagonalNormalEP(HasEntropyEP[MultivariateDiagonalNormalNP],
                                   Multidimensional,
                                   Samplable):
    """The expectation parametrization of the normal distribution with diagonal variance.

    Args:
        mean: E(x).
        second moment: The diagonal elements of E(x^2).
    """
    mean: JaxRealArray = distribution_parameter(VectorSupport())
    second_moment: JaxRealArray = distribution_parameter(VectorSupport(ring=positive_support))

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
        xp = self.array_namespace()
        return xp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_variance_parametrization().sample(key, shape)

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    def variance(self) -> JaxRealArray:
        xp = self.array_namespace()
        return self.second_moment - xp.square(self.mean)

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
        shape = self.mean.shape if shape is None else shape + self.mean.shape
        xp = self.array_namespace()
        deviation = xp.sqrt(self.variance)
        return jr.normal(key, shape) * deviation + self.mean

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    def pdf(self, x: JaxRealArray) -> JaxRealArray:
        return self.to_nat().pdf(x)

    def to_exp(self) -> MultivariateDiagonalNormalEP:
        xp = self.array_namespace()
        second_moment = self.variance + xp.square(self.mean)
        return MultivariateDiagonalNormalEP(self.mean, second_moment)

    def to_nat(self) -> MultivariateDiagonalNormalNP:
        xp = self.array_namespace()
        precision = xp.reciprocal(self.variance)
        mean_times_precision = self.mean * precision
        negative_half_precision = -0.5 * precision
        return MultivariateDiagonalNormalNP(mean_times_precision, negative_half_precision)
