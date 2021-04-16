from __future__ import annotations

from typing import Type

import jax.numpy as jnp
from tjax import IntegralArray, RealArray
from tjax.dataclasses import dataclass

from ..parameter import ScalarSupport, distribution_parameter
from .negative_binomial_common import NBCommonEP, NBCommonNP

__all__ = ['NegativeBinomialNP', 'NegativeBinomialEP']


@dataclass
class NegativeBinomialNP(NBCommonNP['NegativeBinomialEP']):
    failures: IntegralArray = distribution_parameter(ScalarSupport(), fixed=True)
    log_not_p: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    def to_exp(self) -> NegativeBinomialEP:
        return NegativeBinomialEP(self.failures, self._mean())

    def sufficient_statistics(self, x: RealArray) -> NegativeBinomialEP:
        return NegativeBinomialEP(jnp.broadcast_to(self.failures, x.shape), x)


@dataclass
class NegativeBinomialEP(NBCommonEP[NegativeBinomialNP]):
    failures: IntegralArray = distribution_parameter(ScalarSupport(), fixed=True)
    mean: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @classmethod
    def natural_parametrization_cls(cls) -> Type[NegativeBinomialNP]:
        return NegativeBinomialNP

    def to_nat(self) -> NegativeBinomialNP:
        return NegativeBinomialNP(self.failures, self._log_not_p())
