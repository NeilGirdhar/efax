from __future__ import annotations

from typing import Any, Generic, TypeVar

import jax.numpy as jnp
from jax.scipy.special import gammaln
from tjax import IntegralNumeric, RealArray, Shape
from tjax.dataclasses import dataclass

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import EP, NaturalParametrization

__all__: list[str] = []


@dataclass
class NBCommonNP(NaturalParametrization[EP, RealArray], Generic[EP]):
    log_not_p: RealArray

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.log_not_p.shape

    def log_normalizer(self) -> RealArray:
        return -self._failures() * jnp.log1p(-jnp.exp(self.log_not_p))  # type: ignore[return-value]

    def carrier_measure(self, x: RealArray) -> RealArray:
        a = x + self._failures() - 1
        # Return log(a choose x).
        return gammaln(a + 1) - gammaln(x + 1) - gammaln(a - x + 1)

    # Private methods ------------------------------------------------------------------------------
    def _mean(self) -> RealArray:
        return self._failures() / jnp.expm1(-self.log_not_p)

    # Private abstract methods ---------------------------------------------------------------------
    def _failures(self) -> IntegralNumeric:
        raise NotImplementedError


NP = TypeVar('NP', bound=NBCommonNP[Any])


@dataclass
class NBCommonEP(ExpectationParametrization[NP], Generic[NP]):
    mean: RealArray

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape

    # def conjugate_prior_distribution(self, n: RealArray) -> BetaPrimeNP:
    #     return BetaPrimeNP(n * self._failures() * (self.mean, jnp.ones_like(self.mean)))

    # Private methods ------------------------------------------------------------------------------
    def _log_not_p(self) -> RealArray:
        return -jnp.log1p(self._failures() / self.mean)

    # Private abstract methods ---------------------------------------------------------------------
    def _failures(self) -> IntegralNumeric:
        raise NotImplementedError
