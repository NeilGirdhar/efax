from __future__ import annotations

import jax.numpy as jnp
from tjax import JaxRealArray
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..parameter import ScalarSupport, distribution_parameter
from .negative_binomial_common import NBCommonEP, NBCommonNP

__all__ = ['GeometricNP', 'GeometricEP']


@dataclass
class GeometricNP(NBCommonNP['GeometricEP']):
    log_not_p: JaxRealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @override
    def to_exp(self) -> GeometricEP:
        return GeometricEP(self._mean())

    @override
    def sufficient_statistics(self, x: JaxRealArray) -> GeometricEP:
        return GeometricEP(x)

    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.log_not_p.shape)

    @override
    def _failures(self) -> int:
        return 1


@dataclass
class GeometricEP(NBCommonEP[GeometricNP]):
    mean: JaxRealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[GeometricNP]:
        return GeometricNP

    @override
    def to_nat(self) -> GeometricNP:
        return GeometricNP(self._log_not_p())

    @override
    def _failures(self) -> int:
        return 1
