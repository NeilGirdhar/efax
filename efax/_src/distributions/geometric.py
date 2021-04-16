from __future__ import annotations

from typing import Type

import jax.numpy as jnp
from tjax import RealArray
from tjax.dataclasses import dataclass

from ..parameter import ScalarSupport, distribution_parameter
from .negative_binomial_common import NBCommonEP, NBCommonNP

__all__ = ['GeometricNP', 'GeometricEP']


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
    @classmethod
    def natural_parametrization_cls(cls) -> Type[GeometricNP]:
        return GeometricNP

    def to_nat(self) -> GeometricNP:
        return GeometricNP(self._log_not_p())
