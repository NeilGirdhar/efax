from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from tjax import JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter

__all__ = ['WeibullNP', 'WeibullEP']


@dataclass
class WeibullNP(HasEntropyNP, NaturalParametrization['WeibullEP', JaxRealArray]):
    concentration: JaxRealArray = distribution_parameter(ScalarSupport(), fixed=True)
    # eta = -scale^-concentration
    eta: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.eta.shape

    @override
    def domain_support(self) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        return -jnp.log(-self.eta) - jnp.log(self.concentration)

    @override
    def to_exp(self) -> WeibullEP:
        return WeibullEP(self.concentration, -1.0 / self.eta)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return (self.concentration - 1.0) * jnp.log(x)

    @override
    def sufficient_statistics(self, x: JaxRealArray) -> WeibullEP:
        return WeibullEP(jnp.broadcast_to(self.concentration, x.shape), x ** self.concentration)


@dataclass
class WeibullEP(HasEntropyEP[WeibullNP], ExpectationParametrization[WeibullNP], Samplable):
    concentration: JaxRealArray = distribution_parameter(ScalarSupport(), fixed=True)
    # chi = scale^concentration
    chi: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.chi.shape

    @override
    def domain_support(self) -> ScalarSupport:
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
        k = self.concentration
        one_minus_one_over_k = 1.0 - 1.0 / k
        return one_minus_one_over_k * jnp.log(self.chi) - jnp.euler_gamma * one_minus_one_over_k

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        lambda_ = self.chi ** (1.0 / self.concentration)
        return jax.random.weibull_min(key, lambda_, self.concentration, shape)
