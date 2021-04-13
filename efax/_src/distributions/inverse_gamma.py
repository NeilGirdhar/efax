from __future__ import annotations

from typing_extensions import override
import jax.numpy as jnp
from jax.random import KeyArray
from tjax import Array, JaxRealArray, Shape
from tjax.dataclasses import dataclass

from ..parameter import ScalarSupport, distribution_parameter
from ..samplable import Samplable
from ..transformed_parametrization import (TransformedExpectationParametrization,
                                           TransformedNaturalParametrization)
from .gamma import GammaEP, GammaNP

__all__ = ['InverseGammaNP', 'InverseGammaEP']


@dataclass
class InverseGammaNP(TransformedNaturalParametrization[GammaNP, GammaEP, 'InverseGammaEP',
                                                       JaxRealArray],
                     Samplable):
    negative_scale: JaxRealArray = distribution_parameter(ScalarSupport())
    negative_shape_minus_one: JaxRealArray = distribution_parameter(ScalarSupport())

    @override
    def base_distribution(self) -> GammaNP:
        return GammaNP(self.negative_scale, -self.negative_shape_minus_one - 2.0)

    @override
    def create_expectation(self, expectation_parametrization: GammaEP) -> InverseGammaEP:
        return InverseGammaEP(expectation_parametrization.mean,
                              expectation_parametrization.mean_log)

    @override
    def sample_to_base_sample(self, x: Array) -> JaxRealArray:
        return jnp.asarray(1.0 / x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        y = self.base_distribution().sample(key, shape)
        return 1.0 / y  # Convert base sample to sample.

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape)


@dataclass
class InverseGammaEP(TransformedExpectationParametrization[GammaEP, GammaNP, InverseGammaNP]):
    mean_reciprocal: JaxRealArray = distribution_parameter(ScalarSupport())
    negative_mean_log: JaxRealArray = distribution_parameter(ScalarSupport())

    @override
    def base_distribution(self) -> GammaEP:
        return GammaEP(self.mean_reciprocal, self.negative_mean_log)

    @override
    def create_natural(self, natural_parametrization: GammaNP) -> InverseGammaNP:
        return InverseGammaNP(natural_parametrization.negative_rate,
                              -natural_parametrization.shape_minus_one - 2.0)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.shape)
