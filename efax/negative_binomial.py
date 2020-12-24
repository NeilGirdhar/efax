from __future__ import annotations

from typing import Generic

from jax import numpy as jnp
from jax.scipy.special import gammaln
from tjax import RealArray, Shape, dataclass, field

from .exponential_family import EP, NP, ExpectationParametrization, NaturalParametrization
from .parameter import distribution_parameter

__all__ = ['NegativeBinomialNP', 'NegativeBinomialEP', 'GeometricNP', 'GeometricEP']


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


class NBCommonEP(ExpectationParametrization[NP], Generic[NP]):
    failures: int
    mean: RealArray

    # Implemented methods --------------------------------------------------------------------------
    def expected_carrier_measure(self) -> RealArray:
        if self.failures == 1:
            shape = self.mean.shape
            return jnp.zeros(shape)
        raise NotImplementedError

    # Private methods ------------------------------------------------------------------------------
    def _log_not_p(self) -> NegativeBinomialNP:
        return -jnp.log1p(self.failures / self.mean)

    # Overridden methods ---------------------------------------------------------------------------
    # def conjugate_prior_distribution(self, n: RealArray) -> BetaPrimeNP:
    #     return BetaPrimeNP(n * self.failures * (self.mean, jnp.ones_like(self.mean)))


@dataclass
class NegativeBinomialNP(NBCommonNP['NegativeBinomialEP']):
    failures: int = field(static=True)
    log_not_p: RealArray = distribution_parameter(axes=0)

    def to_exp(self) -> NegativeBinomialEP:
        return NegativeBinomialEP(self.failures, self._mean())

    def sufficient_statistics(self, x: RealArray) -> NegativeBinomialEP:
        return NegativeBinomialEP(self.failures, x)


@dataclass
class NegativeBinomialEP(NBCommonEP[NegativeBinomialNP]):
    failures: int = field(static=True)
    mean: RealArray = distribution_parameter(axes=0)

    def to_nat(self) -> NegativeBinomialNP:
        return NegativeBinomialNP(self.failures, self._log_not_p())


@dataclass
class GeometricNP(NBCommonNP['GeometricEP']):
    log_not_p: RealArray = distribution_parameter(axes=0)
    failures = 1

    def to_exp(self) -> GeometricEP:
        return GeometricEP(self._mean())

    def sufficient_statistics(self, x: RealArray) -> GeometricEP:
        return GeometricEP(x)


@dataclass
class GeometricEP(NBCommonEP[GeometricNP]):
    mean: RealArray = distribution_parameter(axes=0)
    failures = 1

    def to_nat(self) -> GeometricNP:
        return GeometricNP(self._log_not_p())
