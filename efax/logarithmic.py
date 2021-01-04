from __future__ import annotations

from typing import Type

from jax import numpy as jnp
from jax.nn import softplus
from tjax import RealArray, Shape, dataclass

from .exp_to_nat import ExpToNat
from .natural_parametrization import NaturalParametrization
from .parameter import distribution_parameter

__all__ = ['LogarithmicNP', 'LogarithmicEP']


@dataclass
class LogarithmicNP(NaturalParametrization['LogarithmicEP']):
    log_probability: RealArray = distribution_parameter(axes=0)

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.log_probability.shape

    def log_normalizer(self) -> RealArray:
        return jnp.log(-jnp.log1p(-jnp.exp(self.log_probability)))

    def to_exp(self) -> LogarithmicEP:
        probability = jnp.exp(self.log_probability)
        return LogarithmicEP(jnp.where(self.log_probability < -100,
                                       1.0,
                                       probability / (jnp.expm1(self.log_probability)
                                                      * jnp.log1p(-probability))))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return -jnp.log(x)

    def sufficient_statistics(self, x: RealArray) -> LogarithmicEP:
        return LogarithmicEP(x)


@dataclass
class LogarithmicEP(ExpToNat[LogarithmicNP]):
    chi: RealArray = distribution_parameter(axes=0)  # - odds / log(1-p)

    # Implemented methods --------------------------------------------------------------------------
    @classmethod
    def natural_parametrization_cls(cls) -> Type[LogarithmicNP]:
        return LogarithmicNP

    def shape(self) -> Shape:
        return self.chi.shape

    # The expected_carrier_measure is unknown.

    @classmethod
    def transform_natural_for_iteration(cls, iteration_natural: LogarithmicNP) -> LogarithmicNP:
        # Run Newton's method on the whole real line.
        return LogarithmicNP(-softplus(-iteration_natural.log_probability))

    # Overridden methods ---------------------------------------------------------------------------
    def to_nat(self) -> LogarithmicNP:
        z = super().to_nat()
        return LogarithmicNP(jnp.where(self.chi < 1.0,
                                       jnp.nan,
                                       jnp.where(self.chi == 1.0,
                                                 jnp.inf,
                                                 z.log_probability)))
