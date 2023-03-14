from __future__ import annotations

import jax.numpy as jnp
from jax.nn import softplus
from tjax import JaxRealArray, Shape
from tjax.dataclasses import dataclass

from ..exp_to_nat import ExpToNat
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter

__all__ = ['LogarithmicNP', 'LogarithmicEP']


log_probability_floor = -50.0
log_probability_ceiling = -1e-7


@dataclass
class LogarithmicNP(NaturalParametrization['LogarithmicEP', JaxRealArray]):
    log_probability: JaxRealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.log_probability.shape

    def log_normalizer(self) -> JaxRealArray:
        return jnp.log(-jnp.log1p(-jnp.exp(self.log_probability)))

    def to_exp(self) -> LogarithmicEP:
        probability = jnp.exp(self.log_probability)
        chi = jnp.where(self.log_probability < log_probability_floor, 1.0,
                        jnp.where(self.log_probability > log_probability_ceiling, jnp.inf,
                                  probability / (jnp.expm1(self.log_probability)
                                                 * jnp.log1p(-probability))))
        return LogarithmicEP(chi)

    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return -jnp.log(x)

    def sufficient_statistics(self, x: JaxRealArray) -> LogarithmicEP:
        return LogarithmicEP(x)


@dataclass
class LogarithmicEP(ExpToNat[LogarithmicNP, JaxRealArray]):
    chi: JaxRealArray = distribution_parameter(ScalarSupport())  # - odds / log(1-p)

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.chi.shape

    @classmethod
    def natural_parametrization_cls(cls) -> type[LogarithmicNP]:
        return LogarithmicNP

    # The expected_carrier_measure is unknown.

    def initial_search_parameters(self) -> JaxRealArray:
        return jnp.zeros(self.chi.shape)

    def search_to_natural(self, search_parameters: JaxRealArray) -> LogarithmicNP:
        # Run Newton's method on the whole real line.
        return LogarithmicNP(-softplus(-search_parameters))

    def search_gradient(self, search_parameters: JaxRealArray) -> JaxRealArray:
        return self._natural_gradient(self.search_to_natural(search_parameters)).log_probability

    # Overridden methods ---------------------------------------------------------------------------
    def to_nat(self) -> LogarithmicNP:
        z: LogarithmicNP = super().to_nat()
        return LogarithmicNP(jnp.where(self.chi < 1.0,  # noqa: PLR2004
                                       jnp.nan,
                                       jnp.where(self.chi == 1.0,  # noqa: PLR2004
                                                 jnp.inf,
                                                 z.log_probability)))
