from __future__ import annotations

from typing import Any, Generic, List, TypeVar

from jax import numpy as jnp
from jax.scipy.special import gammaln
from tjax import RealArray, Shape

from .expectation_parametrization import ExpectationParametrization
from .natural_parametrization import NaturalParametrization

__all__: List[str] = []


EP = TypeVar('EP', bound='NBCommonEP[Any]')


class NBCommonNP(NaturalParametrization[EP], Generic[EP]):
    failures: int
    log_not_p: RealArray

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.log_not_p.shape

    def log_normalizer(self) -> RealArray:
        return -self.failures * jnp.log1p(-jnp.exp(self.log_not_p))

    def carrier_measure(self, x: RealArray) -> RealArray:
        a = x + self.failures - 1
        # Return log(a choose x).
        return gammaln(a + 1) - gammaln(x + 1) - gammaln(a - x + 1)

    # Private methods ------------------------------------------------------------------------------
    def _mean(self) -> RealArray:
        return self.failures / jnp.expm1(-self.log_not_p)


NP = TypeVar('NP', bound=NBCommonNP[Any])


class NBCommonEP(ExpectationParametrization[NP], Generic[NP]):
    failures: int
    mean: RealArray

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean.shape

    def expected_carrier_measure(self) -> RealArray:
        if self.failures == 1:
            shape = self.mean.shape
            return jnp.zeros(shape)
        raise NotImplementedError

    # Private methods ------------------------------------------------------------------------------
    def _log_not_p(self) -> RealArray:
        return -jnp.log1p(self.failures / self.mean)

    # Overridden methods ---------------------------------------------------------------------------
    # def conjugate_prior_distribution(self, n: RealArray) -> BetaPrimeNP:
    #     return BetaPrimeNP(n * self.failures * (self.mean, jnp.ones_like(self.mean)))
