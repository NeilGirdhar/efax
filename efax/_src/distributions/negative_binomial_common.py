from __future__ import annotations

from abc import abstractmethod
from typing import Any, TypeVar

import jax.scipy.special as jss
from tjax import JaxIntegralArray, JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, integral_ring
from ..parametrization import SimpleDistribution

EP = TypeVar('EP', bound='NBCommonEP[Any]')


@dataclass
class NBCommonNP(NaturalParametrization[EP, JaxRealArray],
                 SimpleDistribution):
    log_not_p: JaxRealArray

    @property
    @override
    def shape(self) -> Shape:
        return self.log_not_p.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=integral_ring)

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.array_namespace()
        return -self._failures() * xp.log1p(-xp.exp(self.log_not_p))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        a = x + self._failures() - 1
        # Return log(a choose x).
        return jss.gammaln(a + 1) - jss.gammaln(x + 1) - jss.gammaln(a - x + 1)

    def _mean(self) -> JaxRealArray:
        xp = self.array_namespace()
        return self._failures() / xp.expm1(-self.log_not_p)

    @abstractmethod
    def _failures(self) -> int | JaxIntegralArray:
        raise NotImplementedError


NP = TypeVar('NP', bound=NBCommonNP[Any])


@dataclass
class NBCommonEP(ExpectationParametrization[NP],
                 SimpleDistribution):
    mean: JaxRealArray

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=integral_ring)

    # def conjugate_prior_distribution(self, n: JaxRealArray) -> BetaPrimeNP:
    #     return BetaPrimeNP(n * self._failures() * (self.mean, xp.ones_like(self.mean)))

    def _log_not_p(self) -> JaxRealArray:
        xp = self.array_namespace()
        return -xp.log1p(self._failures() / self.mean)

    @abstractmethod
    def _failures(self) -> int | JaxIntegralArray:
        raise NotImplementedError
