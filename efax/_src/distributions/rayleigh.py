from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from tjax import Array, JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..mixins.transformed_parametrization import (TransformedExpectationParametrization,
                                                  TransformedNaturalParametrization)
from ..parameter import ScalarSupport, distribution_parameter
from .exponential import ExponentialEP, ExponentialNP

__all__ = ['RayleighNP', 'RayleighEP']


@dataclass
class RayleighNP(HasEntropyNP['RayleighEP'],
                 TransformedNaturalParametrization[ExponentialNP, ExponentialEP, 'RayleighEP',
                                                   JaxRealArray]):
    """The natural parametrization of the Rayleigh distribution.

    Args:
        eta: -1 / (2 * sigma^2).
    """
    eta: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.eta.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    @classmethod
    def base_distribution_cls(cls) -> type[ExponentialNP]:
        return ExponentialNP

    @override
    def base_distribution(self) -> ExponentialNP:
        return ExponentialNP(self.eta)

    @override
    @classmethod
    def create_expectation(cls, expectation_parametrization: ExponentialEP) -> RayleighEP:
        return RayleighEP(expectation_parametrization.mean)

    @override
    @classmethod
    def sample_to_base_sample(cls, x: Array, **fixed_parameters: Any
                              ) -> JaxRealArray:
        return jnp.square(x)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.log(x) + jnp.log(2)


@dataclass
class RayleighEP(HasEntropyEP[RayleighNP],
                 TransformedExpectationParametrization[ExponentialEP, ExponentialNP, RayleighNP]):
    """The expectation parametrization of the Rayleigh distribution.

    Args:
        chi: 2 * sigma^2.
    """
    chi: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.chi.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[RayleighNP]:
        return RayleighNP

    @override
    @classmethod
    def base_distribution_cls(cls) -> type[ExponentialEP]:
        return ExponentialEP

    @override
    def base_distribution(self) -> ExponentialEP:
        return ExponentialEP(self.chi)

    @override
    def create_natural(self, natural_parametrization: ExponentialNP) -> RayleighNP:
        return RayleighNP(natural_parametrization.negative_rate)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return 0.5 * jnp.log(self.chi * 0.5) + (1.5 * jnp.log(2.0) - 0.5 * jnp.euler_gamma)
