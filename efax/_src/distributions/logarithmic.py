from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax.nn import softplus
from tjax import JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..mixins.exp_to_nat import ExpToNat
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter

__all__ = ['LogarithmicNP', 'LogarithmicEP']


log_probability_floor = -50.0
log_probability_ceiling = -1e-7


@dataclass
class LogarithmicNP(NaturalParametrization['LogarithmicEP', JaxRealArray]):
    """The natural parametrization of the logarithmic distribution.

    Args:
        log_probability: log(p).
    """
    log_probability: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.log_probability.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        return jnp.log(-jnp.log1p(-jnp.exp(self.log_probability)))

    @override
    def to_exp(self) -> LogarithmicEP:
        probability = jnp.exp(self.log_probability)
        chi = jnp.where(self.log_probability < log_probability_floor, 1.0,
                        jnp.where(self.log_probability > log_probability_ceiling, jnp.inf,
                                  probability / (jnp.expm1(self.log_probability)
                                                 * jnp.log1p(-probability))))
        return LogarithmicEP(chi)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return -jnp.log(x)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> LogarithmicEP:
        return LogarithmicEP(x)


@dataclass
class LogarithmicEP(ExpToNat[LogarithmicNP, JaxRealArray],
                    ExpectationParametrization[LogarithmicNP]):
    """The expectation parametrization of the logarithmic distribution.

    Args:
        chi: -(p / (1-p)) * log(1-p).
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
    def natural_parametrization_cls(cls) -> type[LogarithmicNP]:
        return LogarithmicNP

    # The expected_carrier_measure is unknown.

    @override
    def initial_search_parameters(self) -> JaxRealArray:
        return jnp.zeros(self.chi.shape)

    @override
    def search_to_natural(self, search_parameters: JaxRealArray) -> LogarithmicNP:
        # Run Newton's method on the whole real line.
        return LogarithmicNP(-softplus(-search_parameters))

    @override
    def search_gradient(self, search_parameters: JaxRealArray) -> JaxRealArray:
        return self._natural_gradient(self.search_to_natural(search_parameters)).log_probability

    @override
    def to_nat(self) -> LogarithmicNP:
        z: LogarithmicNP = super().to_nat()
        return LogarithmicNP(jnp.where(self.chi < 1.0,  # noqa: PLR2004
                                       jnp.nan,
                                       jnp.where(self.chi == 1.0,  # noqa: PLR2004
                                                 jnp.inf,
                                                 z.log_probability)))
