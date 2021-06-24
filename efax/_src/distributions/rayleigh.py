from __future__ import annotations

from typing import Type

import jax.numpy as jnp
from tjax import Array, RealArray
from tjax.dataclasses import dataclass

from ..parameter import ScalarSupport, distribution_parameter
from ..transformed_parametrization import (TransformedExpectationParametrization,
                                           TransformedNaturalParametrization)
from .exponential import ExponentialEP, ExponentialNP

__all__ = ['RayleighNP', 'RayleighEP']


@dataclass
class RayleighNP(TransformedNaturalParametrization[ExponentialNP, ExponentialEP, 'RayleighEP',
                                                   RealArray]):
    # eta = -1 / (2 * sigma^2)
    eta: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    def base_distribution(self) -> ExponentialNP:
        return ExponentialNP(self.eta)

    def create_expectation(self, expectation_parametrization: ExponentialEP) -> RayleighEP:
        return RayleighEP(expectation_parametrization.mean)

    def sample_to_base_sample(self, x: Array) -> RealArray:
        return jnp.square(x)

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.log(x) + jnp.log(2)


@dataclass
class RayleighEP(TransformedExpectationParametrization[ExponentialEP, ExponentialNP, RayleighNP]):
    # chi = 2 * sigma^2
    chi: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @classmethod
    def natural_parametrization_cls(cls) -> Type[RayleighNP]:
        return RayleighNP

    def base_distribution(self) -> ExponentialEP:
        return ExponentialEP(self.chi)

    def create_natural(self, natural_parametrization: ExponentialNP) -> RayleighNP:
        return RayleighNP(natural_parametrization.negative_rate)

    def expected_carrier_measure(self) -> RealArray:
        return 0.5 * jnp.log(self.chi * 0.5) + (1.5 * jnp.log(2.0) - 0.5 * jnp.euler_gamma)
