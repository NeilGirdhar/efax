from __future__ import annotations

import math
from typing import Any

from array_api_compat import get_namespace
import jax
from tjax import KeyArray
from tjax import JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.conjugate_prior import HasGeneralizedConjugatePrior
from ..interfaces.multidimensional import Multidimensional
from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, SymmetricMatrixSupport, VectorSupport, distribution_parameter
from .multivariate_normal.diagonal import MultivariateDiagonalNormalNP
from .multivariate_normal.isotropic import IsotropicNormalNP


@dataclass
class WishartNP(HasEntropyNP['WishartEP'],
                NaturalParametrization['WishartEP', JaxRealArray],
                Multidimensional,
                Samplable):
    """The multivariate normal distribution with fixed variance.

    This is a curved exponential family.
    """
    mean_times_precision: JaxRealArray = distribution_parameter(VectorSupport())
    variance: JaxRealArray = distribution_parameter(ScalarSupport(), fixed=True)

    # Implemented methods --------------------------------------------------------------------------
    @property
    @override
    def shape(self) -> Shape:
        return self.mean_times_precision.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> SymmetricMatrixSupport:
        return SymmetricMatrixSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.get_namespace()
        eta = self.mean_times_precision
        return 0.5 * (xp.sum(xp.square(eta), axis=-1) * self.variance
                      + self.dimensions() * xp.log(math.pi * 2.0 * self.variance))

    @override
    def to_exp(self) -> WishartEP:
        xp = self.get_namespace()
        return WishartEP(
            self.mean_times_precision * self.variance[..., xp.newaxis],
            variance=self.variance)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.get_namespace()
        return -0.5 * xp.sum(xp.square(x), axis=-1) / self.variance

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> WishartEP:
        xp = get_namespace(x)
        shape = x.shape[:-1]
        variance = xp.broadcast_to(fixed_parameters['variance'], shape)
        return WishartEP(x, variance=variance)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)

    @override
    def dimensions(self) -> int:
        return self.mean_times_precision.shape[-1]


@dataclass
class WishartEP(
        HasEntropyEP[WishartNP],
        HasGeneralizedConjugatePrior,
        Multidimensional,
        Samplable):
    mean: JaxRealArray = distribution_parameter(VectorSupport())
    variance: JaxRealArray = distribution_parameter(ScalarSupport(), fixed=True)

    # Implemented methods --------------------------------------------------------------------------
    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> SymmetricMatrixSupport:
        return SymmetricMatrixSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[WishartNP]:
        return WishartNP

    @override
    def to_nat(self) -> WishartNP:
        xp = self.get_namespace()
        return WishartNP(self.mean / self.variance[..., xp.newaxis],
                                                 variance=self.variance)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.get_namespace()
        return -0.5 * (xp.sum(xp.square(self.mean), axis=-1) / self.variance + self.dimensions())

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        xp = self.get_namespace()
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        variance = self.variance[..., xp.newaxis]
        deviation = xp.sqrt(variance)
        return jax.random.normal(key, shape) * deviation + self.mean

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> IsotropicNormalNP:
        xp = self.get_namespace()
        negative_half_precision = -0.5 * n / self.variance
        return IsotropicNormalNP(n[..., xp.newaxis] * self.mean, negative_half_precision)

    @override
    def generalized_conjugate_prior_distribution(self, n: JaxRealArray
                                                 ) -> MultivariateDiagonalNormalNP:
        xp = self.get_namespace()
        negative_half_precision = -0.5 * n / self.variance[..., xp.newaxis]
        return MultivariateDiagonalNormalNP(n * self.mean, negative_half_precision)

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.mean

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]
