from __future__ import annotations

import jax.random as jr
import numpy as np
from array_api_compat import array_namespace
from tjax import JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter, negative_support, positive_support
from ..parametrization import SimpleDistribution


@dataclass
class WeibullNP(HasEntropyNP['WeibullEP'],
                Samplable,
                NaturalParametrization['WeibullEP', JaxRealArray],
                SimpleDistribution):
    """The natural parametrization of the Weibull distribution.

    Args:
        concentration: The fixed shape parameter k.
        eta: -scale ^ -concentration.
    """
    concentration: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support),
                                                         fixed=True)
    eta: JaxRealArray = distribution_parameter(ScalarSupport(ring=negative_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.eta.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.array_namespace()
        return -xp.log(-self.eta) - xp.log(self.concentration)

    @override
    def to_exp(self) -> WeibullEP:
        xp = self.array_namespace()
        return WeibullEP(self.concentration, -xp.reciprocal(self.eta))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return (self.concentration - 1.0) * xp.log(x)

    @override
    @classmethod
    def sufficient_statistics(cls,
                              x: JaxRealArray,
                              **fixed_parameters: JaxArray
                              ) -> WeibullEP:
        xp = array_namespace(x)
        concentration = fixed_parameters['concentration']
        return WeibullEP(xp.broadcast_to(concentration, x.shape), x ** concentration)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)


@dataclass
class WeibullEP(HasEntropyEP[WeibullNP],
                Samplable,
                SimpleDistribution):
    """The expectation parametrization of the Weibull distribution.

    Args:
        concentration: The fixed shape parameter k.
        chi: scale ^ concentration.
    """
    concentration: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support),
                                                         fixed=True)
    chi: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

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
    def natural_parametrization_cls(cls) -> type[WeibullNP]:
        return WeibullNP

    @override
    def to_nat(self) -> WeibullNP:
        xp = self.array_namespace()
        return WeibullNP(self.concentration, -xp.reciprocal(self.chi))

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.array_namespace()
        k = self.concentration
        one_minus_one_over_k = 1.0 - xp.reciprocal(k)
        return one_minus_one_over_k * xp.log(self.chi) - np.euler_gamma * one_minus_one_over_k

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        shape = self.shape if shape is None else shape + self.shape
        xp = self.array_namespace()
        lambda_ = self.chi ** xp.reciprocal(self.concentration)
        return jr.weibull_min(key, lambda_, self.concentration, shape)
