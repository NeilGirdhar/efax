from __future__ import annotations

import math
from typing import Iterable, Optional

from chex import Array
from jax import numpy as jnp
from jax.nn import softplus
from tjax import RealArray, Shape, dataclass

from .exp_to_nat import ExpToNat
from .exponential_family import NaturalParametrization

__all__ = ['LogarithmicNP', 'LogarithmicEP']


@dataclass
class LogarithmicNP(NaturalParametrization['LogarithmicEP']):
    log_probability: RealArray

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

    @classmethod
    def field_axes(cls) -> Iterable[int]:
        yield 0


@dataclass
class LogarithmicEP(ExpToNat[LogarithmicNP]):
    chi: RealArray  # - odds / log(1-p)

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.chi.shape

    # The expected_carrier_measure is unknown.

    # Overridden methods ---------------------------------------------------------------------------
    def to_nat_early_out(self) -> Optional[Array]:
        if self.chi < 1.0:
            raise ValueError
        if self.chi == 1.0:
            return math.inf
        return None

    @classmethod
    def unflatten_natural(cls, flattened_natural: Array) -> LogarithmicNP:
        # Run Newton's method on the whole real line.
        return LogarithmicNP(-softplus(flattened_natural))

    def flatten_expectation(self) -> Array:
        return self.chi
