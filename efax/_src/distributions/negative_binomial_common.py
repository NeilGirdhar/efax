from __future__ import annotations

from typing import Any, Generic, TypeVar

import jax.numpy as jnp
from jax.scipy.special import gammaln
from tjax import JaxIntegralArray, JaxRealArray, Shape
from tjax.dataclasses import dataclass

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import EP, NaturalParametrization

__all__: list[str] = []


@dataclass
class NBCommonNP(NaturalParametrization[EP, JaxRealArray], Generic[EP]):
    log_not_p: JaxRealArray

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.log_not_p.shape

    def log_normalizer(self) -> JaxRealArray:
        return -self._failures() * jnp.log1p(-jnp.exp(self.log_not_p))  # type: ignore[return-value]

    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        a = x + self._failures() - 1
        # Return log(a choose x).
        return gammaln(a + 1) - gammaln(x + 1) - gammaln(a - x + 1)

    # Private methods ------------------------------------------------------------------------------
    def _mean(self) -> JaxRealArray:
        return self._failures() / jnp.expm1(-self.log_not_p)  # type: ignore[return-value]

    # Private abstract methods ---------------------------------------------------------------------
    def _failures(self) -> int | JaxIntegralArray:
        raise NotImplementedError


NP = TypeVar('NP', bound=NBCommonNP[Any])


@dataclass
class NBCommonEP(ExpectationParametrization[NP], Generic[NP]):
    mean: JaxRealArray

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape

    # def conjugate_prior_distribution(self, n: JaxRealArray) -> BetaPrimeNP:
    #     return BetaPrimeNP(n * self._failures() * (self.mean, jnp.ones_like(self.mean)))

    # Private methods ------------------------------------------------------------------------------
    def _log_not_p(self) -> JaxRealArray:
        return -jnp.log1p(self._failures() / self.mean)

    # Private abstract methods ---------------------------------------------------------------------
    def _failures(self) -> int | JaxIntegralArray:
        raise NotImplementedError
