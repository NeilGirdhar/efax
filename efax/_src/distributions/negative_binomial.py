from __future__ import annotations

import jax.numpy as jnp
from tjax import IntegerArray, RealArray
from tjax.dataclasses import dataclass

from ..parameter import ScalarSupport, distribution_parameter
from .negative_binomial_common import NBCommonEP, NBCommonNP

__all__ = ['NegativeBinomialNP', 'NegativeBinomialEP', 'GeometricNP', 'GeometricEP']


@dataclass
class NegativeBinomialNP(NBCommonNP['NegativeBinomialEP']):
    failures: IntegerArray = distribution_parameter(ScalarSupport(), fixed=True)
    log_not_p: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    def to_exp(self) -> NegativeBinomialEP:
        return NegativeBinomialEP(self.failures, self._mean())

    def sufficient_statistics(self, x: RealArray) -> NegativeBinomialEP:
        return NegativeBinomialEP(jnp.broadcast_to(self.failures, x.shape), x)


@dataclass
class NegativeBinomialEP(NBCommonEP[NegativeBinomialNP]):
    failures: IntegerArray = distribution_parameter(ScalarSupport(), fixed=True)
    mean: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    def to_nat(self) -> NegativeBinomialNP:
        return NegativeBinomialNP(self.failures, self._log_not_p())


@dataclass
class GeometricNP(NBCommonNP['GeometricEP']):
    log_not_p: RealArray = distribution_parameter(ScalarSupport())
    failures = 1

    # Implemented methods --------------------------------------------------------------------------
    def to_exp(self) -> GeometricEP:
        return GeometricEP(self._mean())

    def sufficient_statistics(self, x: RealArray) -> GeometricEP:
        return GeometricEP(x)

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.log_not_p.shape)


@dataclass
class GeometricEP(NBCommonEP[GeometricNP]):
    mean: RealArray = distribution_parameter(ScalarSupport())
    failures = 1

    # Implemented methods --------------------------------------------------------------------------
    def to_nat(self) -> GeometricNP:
        return GeometricNP(self._log_not_p())
