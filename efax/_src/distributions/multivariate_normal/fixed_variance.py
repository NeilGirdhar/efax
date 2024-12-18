from __future__ import annotations

import math
from typing import Any, Self

import jax
import jax.numpy as jnp
from tjax import JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...expectation_parametrization import ExpectationParametrization
from ...interfaces.conjugate_prior import HasGeneralizedConjugatePrior
from ...interfaces.multidimensional import Multidimensional
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import RealField, ScalarSupport, VectorSupport, distribution_parameter
from .diagonal import MultivariateDiagonalNormalNP
from .isotropic import IsotropicNormalNP


@dataclass
class MultivariateFixedVarianceNormalNP(HasEntropyNP['MultivariateFixedVarianceNormalEP'],
                                        NaturalParametrization['MultivariateFixedVarianceNormalEP',
                                                               JaxRealArray],
                                        Multidimensional,
                                        Samplable):
    """The natural parametrization of the multivariate normal distribution with fixed variance.

    This is a curved exponential family.

    Args:
        mean_times_precision: E(x) / Var(x).
        variance: The fixed variance, Var(x).
    """
    mean_times_precision: JaxRealArray = distribution_parameter(VectorSupport())
    variance: JaxRealArray = distribution_parameter(ScalarSupport(ring=RealField(minimum=0.0)),
                                                    fixed=True)

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
        return 0.5 * (jnp.sum(jnp.square(eta), axis=-1) * self.variance
                      + self.dimensions() * jnp.log(math.pi * 2.0 * self.variance))

    @override
    def to_exp(self) -> MultivariateFixedVarianceNormalEP:
        return MultivariateFixedVarianceNormalEP(
            self.mean_times_precision * self.variance[..., jnp.newaxis],
            variance=self.variance)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return -0.5 * jnp.sum(jnp.square(x), axis=-1) / self.variance

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> MultivariateFixedVarianceNormalEP:
        return MultivariateFixedVarianceNormalEP(x, variance=fixed_parameters['variance'])

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)

    @override
    def dimensions(self) -> int:
        return self.mean_times_precision.shape[-1]


@dataclass
class MultivariateFixedVarianceNormalEP(
        HasEntropyEP[MultivariateFixedVarianceNormalNP],
        HasGeneralizedConjugatePrior,
        Samplable,
        ExpectationParametrization[MultivariateFixedVarianceNormalNP],
        Multidimensional):
    """The expectation parametrization of the multivariate normal distribution with fixed variance.

    This is a curved exponential family.

    Args:
        mean: E(x).
        variance: The fixed variance, Var(x).
    """
    mean: JaxRealArray = distribution_parameter(VectorSupport())
    variance: JaxRealArray = distribution_parameter(ScalarSupport(ring=RealField(minimum=0.0)),
                                                    fixed=True)

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
    def natural_parametrization_cls(cls) -> type[MultivariateFixedVarianceNormalNP]:
        return MultivariateFixedVarianceNormalNP

    @override
    def to_nat(self) -> MultivariateFixedVarianceNormalNP:
        return MultivariateFixedVarianceNormalNP(self.mean / self.variance[..., jnp.newaxis],
                                                 variance=self.variance)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return -0.5 * (jnp.sum(jnp.square(self.mean), axis=-1) / self.variance + self.dimensions())

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        variance = self.variance[..., jnp.newaxis]
        deviation = jnp.sqrt(variance)
        return jax.random.normal(key, shape) * deviation + self.mean

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> IsotropicNormalNP:
        n_over_variance = n / self.variance
        negative_half_precision = -0.5 * n_over_variance
        return IsotropicNormalNP(n_over_variance[..., jnp.newaxis] * self.mean,
                                 negative_half_precision)

    @classmethod
    @override
    def from_conjugate_prior_distribution(cls, cp: NaturalParametrization[Any, Any],
                                          variance: JaxRealArray | None = None
                                          ) -> tuple[Self, JaxRealArray]:
        assert isinstance(cp, IsotropicNormalNP)
        assert variance is not None
        n_over_variance = -2.0 * cp.negative_half_precision
        n = n_over_variance * variance
        mean = cp.mean_times_precision / n_over_variance[..., jnp.newaxis]
        return cls(mean, variance), n

    @override
    def generalized_conjugate_prior_distribution(self, n: JaxRealArray
                                                 ) -> MultivariateDiagonalNormalNP:
        n_over_variance = n / self.variance[..., jnp.newaxis]
        negative_half_precision = -0.5 * n_over_variance
        return MultivariateDiagonalNormalNP(n_over_variance * self.mean, negative_half_precision)

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.mean

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]
