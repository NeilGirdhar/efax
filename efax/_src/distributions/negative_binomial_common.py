from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

import jax.numpy as jnp
from jax.scipy.special import gammaln
from tjax import JaxIntegralArray, JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import EP, NaturalParametrization
from ..parameter import ScalarSupport

__all__: list[str] = []


@dataclass
class NBCommonNP(NaturalParametrization[EP, JaxRealArray], Generic[EP]):
    log_not_p: JaxRealArray

    @property
    @override
    def shape(self) -> Shape:
        return self.log_not_p.shape

    @override
    def domain_support(self) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        return -self._failures() * jnp.log1p(-jnp.exp(self.log_not_p))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        a = x + self._failures() - 1
        # Return log(a choose x).
        return gammaln(a + 1) - gammaln(x + 1) - gammaln(a - x + 1)

    def _mean(self) -> JaxRealArray:
        return self._failures() / jnp.expm1(-self.log_not_p)

    @abstractmethod
    def _failures(self) -> int | JaxIntegralArray:
        raise NotImplementedError


NP = TypeVar('NP', bound=NBCommonNP[Any])


@dataclass
class NBCommonEP(ExpectationParametrization[NP], Generic[NP]):
    mean: JaxRealArray

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    def domain_support(self) -> ScalarSupport:
        return ScalarSupport()

    # def conjugate_prior_distribution(self, n: JaxRealArray) -> BetaPrimeNP:
    #     return BetaPrimeNP(n * self._failures() * (self.mean, jnp.ones_like(self.mean)))

    def _log_not_p(self) -> JaxRealArray:
        return -jnp.log1p(self._failures() / self.mean)

    @abstractmethod
    def _failures(self) -> int | JaxIntegralArray:
        raise NotImplementedError
