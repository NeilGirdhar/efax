from __future__ import annotations

import jax.numpy as jnp
from tjax import Array, JaxRealArray
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..parameter import ScalarSupport, distribution_parameter
from ..transformed_parametrization import (TransformedExpectationParametrization,
                                           TransformedNaturalParametrization)
from .exponential import ExponentialEP, ExponentialNP

__all__ = ['RayleighNP', 'RayleighEP']


@dataclass
class RayleighNP(TransformedNaturalParametrization[ExponentialNP, ExponentialEP, 'RayleighEP',
                                                   JaxRealArray]):
    # eta = -1 / (2 * sigma^2)
    eta: JaxRealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @override
    def base_distribution(self) -> ExponentialNP:
        return ExponentialNP(self.eta)

    @override
    def create_expectation(self, expectation_parametrization: ExponentialEP) -> RayleighEP:
        return RayleighEP(expectation_parametrization.mean)

    @override
    def sample_to_base_sample(self, x: Array) -> JaxRealArray:
        return jnp.square(x)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.log(x) + jnp.log(2)


@dataclass
class RayleighEP(TransformedExpectationParametrization[ExponentialEP, ExponentialNP, RayleighNP]):
    # chi = 2 * sigma^2
    chi: JaxRealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[RayleighNP]:
        return RayleighNP

    @override
    def base_distribution(self) -> ExponentialEP:
        return ExponentialEP(self.chi)

    @override
    def create_natural(self, natural_parametrization: ExponentialNP) -> RayleighNP:
        return RayleighNP(natural_parametrization.negative_rate)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return 0.5 * jnp.log(self.chi * 0.5) + (1.5 * jnp.log(2.0) - 0.5 * jnp.euler_gamma)
