from __future__ import annotations

import jax.numpy as jnp
from tjax import JaxRealArray
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..parameter import ScalarSupport, distribution_parameter
from .negative_binomial_common import NBCommonEP, NBCommonNP

__all__ = ['GeometricNP', 'GeometricEP']


@dataclass
class GeometricNP(HasEntropyNP,
                  NBCommonNP['GeometricEP']):
    """The natural parameters of the geometric distribution.

    Models the number of Bernoulli trials having probability p until one failures.  Thus, it has
    support {0, ...}.

    Args:
        log_not_p: log(1-p).
    """
    log_not_p: JaxRealArray = distribution_parameter(ScalarSupport())

    @override
    def to_exp(self) -> GeometricEP:
        return GeometricEP(self._mean())

    @override
    def sufficient_statistics(self, x: JaxRealArray) -> GeometricEP:
        return GeometricEP(x)

    @override
    def _failures(self) -> int:
        return 1


@dataclass
class GeometricEP(HasEntropyEP[GeometricNP],
                  NBCommonEP[GeometricNP]):
    """The expectation parameters of the geometric distribution.

    Models the number of Bernoulli trials having probability p until one failures.  Thus, it has
    support {0, ...}.

    Args:
        mean: E(x).
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport())

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[GeometricNP]:
        return GeometricNP

    @override
    def to_nat(self) -> GeometricNP:
        return GeometricNP(self._log_not_p())

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.mean.shape)

    @override
    def _failures(self) -> int:
        return 1
