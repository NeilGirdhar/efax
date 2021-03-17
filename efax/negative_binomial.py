from __future__ import annotations

from tjax import RealArray, dataclass, field

from .negative_binomial_common import NBCommonEP, NBCommonNP
from .parameter import ScalarSupport, distribution_parameter

__all__ = ['NegativeBinomialNP', 'NegativeBinomialEP', 'GeometricNP', 'GeometricEP']


@dataclass
class NegativeBinomialNP(NBCommonNP['NegativeBinomialEP']):
    failures: int = field(static=True)
    log_not_p: RealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    def to_exp(self) -> NegativeBinomialEP:
        return NegativeBinomialEP(self.failures, self._mean())

    def sufficient_statistics(self, x: RealArray) -> NegativeBinomialEP:
        return NegativeBinomialEP(self.failures, x)


@dataclass
class NegativeBinomialEP(NBCommonEP[NegativeBinomialNP]):
    failures: int = field(static=True)
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


@dataclass
class GeometricEP(NBCommonEP[GeometricNP]):
    mean: RealArray = distribution_parameter(ScalarSupport())
    failures = 1

    # Implemented methods --------------------------------------------------------------------------
    def to_nat(self) -> GeometricNP:
        return GeometricNP(self._log_not_p())
