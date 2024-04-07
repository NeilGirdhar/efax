from __future__ import annotations

from typing import Any

import jax
import numpy as np
from array_api_compat import get_namespace
from tjax import JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
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
        xp = self.get_namespace()
        return -xp.log(-self.eta) - xp.log(self.concentration)

    @override
    def to_exp(self) -> WeibullEP:
        return WeibullEP(self.concentration, -1.0 / self.eta)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.get_namespace(x)
        return (self.concentration - 1.0) * xp.log(x)

    @override
    @classmethod
    def sufficient_statistics(cls,
                              x: JaxRealArray,
                              **fixed_parameters: Any
                              ) -> WeibullEP:
        xp = get_namespace(x)
        concentration = fixed_parameters['concentration']
        return WeibullEP(xp.broadcast_to(concentration, x.shape), x ** concentration)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)


@dataclass
class WeibullEP(HasEntropyEP[WeibullNP],
                ExpectationParametrization[WeibullNP],
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
        return WeibullNP(self.concentration, -1.0 / self.chi)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.get_namespace()
        k = self.concentration
        one_minus_one_over_k = 1.0 - 1.0 / k
        return one_minus_one_over_k * xp.log(self.chi) - np.euler_gamma * one_minus_one_over_k

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        lambda_ = self.chi ** (1.0 / self.concentration)
        return jax.random.weibull_min(key, lambda_, self.concentration, shape)
