from __future__ import annotations

import math
from typing import cast

import jax.random as jr
from tjax import (JaxArray, JaxRealArray, KeyArray, Shape, matrix_dot_product, matrix_vector_mul,
                  outer_product)
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...interfaces.multidimensional import Multidimensional
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import SymmetricMatrixSupport, VectorSupport, distribution_parameter


@dataclass
class MultivariateNormalNP(HasEntropyNP['MultivariateNormalEP'],
                           NaturalParametrization['MultivariateNormalEP', JaxRealArray],
                           Multidimensional,
                           Samplable):
    """The natural parametrization of the multivariate normal distribution.

    Args:
        mean_times_precision: E(x) / Var(x).
        negative_half_precision: -0.5 / Var(x).
    """
    mean_times_precision: JaxRealArray = distribution_parameter(VectorSupport())
    negative_half_precision: JaxRealArray = distribution_parameter(SymmetricMatrixSupport(
        negative_semidefinite=True))

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
        eta = self.mean_times_precision
        h_inv = xp.linalg.inv(self.negative_half_precision)
        a = matrix_dot_product(h_inv, outer_product(eta, eta))
        _, ld = xp.linalg.slogdet(-self.negative_half_precision)
        return -0.25 * a - 0.5 * ld + 0.5 * self.dimensions() * math.log(math.pi)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_variance_parametrization().sample(key, shape)

    def variance(self) -> JaxRealArray:
        xp = self.array_namespace()
        h_inv: JaxRealArray = xp.linalg.inv(self.negative_half_precision)
        return -0.5 * h_inv

    def mean(self) -> JaxRealArray:
        variance = self.variance()
        return matrix_vector_mul(variance, self.mean_times_precision)

    def to_variance_parametrization(self) -> MultivariateNormalVP:
        variance = self.variance()
        mean = matrix_vector_mul(variance, self.mean_times_precision)
        return MultivariateNormalVP(mean, variance)

    @override
    def to_exp(self) -> MultivariateNormalEP:
        xp = self.array_namespace()
        h_inv = xp.linalg.inv(self.negative_half_precision)
        h_inv = cast('JaxRealArray', h_inv)
        h_inv_times_eta = matrix_vector_mul(h_inv, self.mean_times_precision)
        mean = -0.5 * h_inv_times_eta
        second_moment = 0.25 * outer_product(h_inv_times_eta, h_inv_times_eta) - 0.5 * h_inv
        return MultivariateNormalEP(mean, second_moment)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return xp.zeros(x.shape[:-1])

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray
                              ) -> MultivariateNormalEP:
        return MultivariateNormalEP(x, outer_product(x, x))

    @override
    def dimensions(self) -> int:
        return self.mean_times_precision.shape[-1]


@dataclass
class MultivariateNormalEP(HasEntropyEP[MultivariateNormalNP],
                           Multidimensional,
                           Samplable):
    mean: JaxRealArray = distribution_parameter(VectorSupport())
    second_moment: JaxRealArray = distribution_parameter(SymmetricMatrixSupport(
        positive_semidefinite=True))

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
        xp = self.array_namespace()
        precision = xp.linalg.inv(self.variance())
        precision = cast('JaxRealArray', precision)
        mean_times_precision = matrix_vector_mul(precision, self.mean)
        return MultivariateNormalNP(mean_times_precision, -0.5 * precision)

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
        return self.second_moment - outer_product(self.mean, self.mean)

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
        shape = self.shape if shape is None else shape + self.shape
        return jr.multivariate_normal(key, self.mean, self.variance, shape)

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    def to_exp(self) -> MultivariateNormalEP:
        second_moment = self.variance + outer_product(self.mean, self.mean)
        return MultivariateNormalEP(self.mean, second_moment)
