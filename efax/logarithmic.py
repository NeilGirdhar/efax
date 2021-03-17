from __future__ import annotations

from typing import Type

import jax.numpy as jnp
from jax.nn import softplus
from tjax import RealArray, Shape, dataclass

from .exp_to_nat import ExpToNat
from .natural_parametrization import NaturalParametrization
from .parameter import ScalarSupport, distribution_parameter

__all__ = ['LogarithmicNP', 'LogarithmicEP']


@dataclass
class LogarithmicNP(NaturalParametrization['LogarithmicEP']):
    log_probability: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.log_probability.shape

    def log_normalizer(self) -> RealArray:
        return jnp.log(-jnp.log1p(-jnp.exp(self.log_probability)))

    def to_exp(self) -> LogarithmicEP:
        probability = jnp.exp(self.log_probability)
        chi = jnp.where(self.log_probability < -50.0, 1.0,
                        jnp.where(self.log_probability > -1e-7, jnp.inf,
                                  probability / (jnp.expm1(self.log_probability)
                                                 * jnp.log1p(-probability))))
        return LogarithmicEP(chi)

    def carrier_measure(self, x: RealArray) -> RealArray:
        return -jnp.log(x)

    def sufficient_statistics(self, x: RealArray) -> LogarithmicEP:
        return LogarithmicEP(x)


@dataclass
class LogarithmicEP(ExpToNat[LogarithmicNP, RealArray]):
    chi: RealArray = distribution_parameter(ScalarSupport())  # - odds / log(1-p)

    # Implemented methods --------------------------------------------------------------------------
    @classmethod
    def natural_parametrization_cls(cls) -> Type[LogarithmicNP]:
        return LogarithmicNP

    def shape(self) -> Shape:
        return self.chi.shape

    # The expected_carrier_measure is unknown.

    def initial_search_parameters(self) -> RealArray:
        return jnp.zeros(self.chi.shape)

    def search_to_natural(self, search_parameters: RealArray) -> LogarithmicNP:
        # Run Newton's method on the whole real line.
        return LogarithmicNP(-softplus(-search_parameters))

    def search_gradient(self, search_parameters: RealArray) -> RealArray:
        return self._natural_gradient(self.search_to_natural(search_parameters)).log_probability

    # Overridden methods ---------------------------------------------------------------------------
    def to_nat(self) -> LogarithmicNP:
        z = super().to_nat()
        return LogarithmicNP(jnp.where(self.chi < 1.0,
                                       jnp.nan,
                                       jnp.where(self.chi == 1.0,
                                                 jnp.inf,
                                                 z.log_probability)))
