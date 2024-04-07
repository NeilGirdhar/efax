from __future__ import annotations

import math
from typing import Any

import numpy as np
from array_api_compat import get_namespace
from jax.random import rayleigh
from tjax import Array, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..mixins.transformed_parametrization import (TransformedExpectationParametrization,
                                                  TransformedNaturalParametrization)
from ..parameter import ScalarSupport, distribution_parameter, negative_support, positive_support
from .exponential import ExponentialEP, ExponentialNP


@dataclass
class RayleighNP(Samplable,
                 HasEntropyNP['RayleighEP'],
                 TransformedNaturalParametrization[ExponentialNP, ExponentialEP, 'RayleighEP',
                                                   JaxRealArray]):
    """The natural parametrization of the Rayleigh distribution.

    Args:
        eta: -1 / (2 * sigma^2).
    """
    eta: JaxRealArray = distribution_parameter(ScalarSupport(ring=negative_support))

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    @classmethod
    def expectation_parametrization_cls(cls) -> type[RayleighEP]:
        return RayleighEP

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
        xp = get_namespace(x)
        return xp.square(x)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.get_namespace(x)
        return xp.log(x) + math.log(2.0)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        xp = self.get_namespace()
        if shape is not None:
            shape += self.shape
        sigma = xp.sqrt(-0.5 / self.eta)
        return rayleigh(key, sigma, shape)


@dataclass
class RayleighEP(Samplable,
                 HasEntropyEP[RayleighNP],
                 TransformedExpectationParametrization[ExponentialEP, ExponentialNP, RayleighNP]):
    """The expectation parametrization of the Rayleigh distribution.

    Args:
        chi: 2 * sigma^2.
    """
    chi: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

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
        xp = self.get_namespace()
        return 0.5 * xp.log(self.chi * 0.5) + (1.5 * math.log(2.0) - 0.5 * np.euler_gamma)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        xp = self.get_namespace()
        if shape is not None:
            shape += self.shape
        sigma = xp.sqrt(0.5 * self.chi)
        return rayleigh(key, sigma, shape)
