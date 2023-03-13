from __future__ import annotations

import jax.numpy as jnp
from tjax import JaxIntegralArray, JaxRealArray
from tjax.dataclasses import dataclass

from ..parameter import ScalarSupport, distribution_parameter
from .negative_binomial_common import NBCommonEP, NBCommonNP

__all__ = ['NegativeBinomialNP', 'NegativeBinomialEP']


@dataclass
class NegativeBinomialNP(NBCommonNP['NegativeBinomialEP']):
    log_not_p: JaxRealArray = distribution_parameter(ScalarSupport())
    failures: JaxIntegralArray = distribution_parameter(ScalarSupport(), fixed=True)

    # Implemented methods --------------------------------------------------------------------------
    def to_exp(self) -> NegativeBinomialEP:
        return NegativeBinomialEP(self._mean(), self.failures)

    def sufficient_statistics(self, x: JaxRealArray) -> NegativeBinomialEP:
        return NegativeBinomialEP(x, jnp.broadcast_to(self.failures, x.shape))

    def _failures(self) -> JaxIntegralArray:
        return self.failures


@dataclass
class NegativeBinomialEP(NBCommonEP[NegativeBinomialNP]):
    mean: JaxRealArray = distribution_parameter(ScalarSupport())
    failures: JaxIntegralArray = distribution_parameter(ScalarSupport(), fixed=True)

    # Implemented methods --------------------------------------------------------------------------
    @classmethod
    def natural_parametrization_cls(cls) -> type[NegativeBinomialNP]:
        return NegativeBinomialNP

    def to_nat(self) -> NegativeBinomialNP:
        return NegativeBinomialNP(self._log_not_p(), self.failures)

    def _failures(self) -> JaxIntegralArray:
        return self.failures
