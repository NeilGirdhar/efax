from __future__ import annotations

from tjax import JaxArray, JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..mixins.exp_to_nat.exp_to_nat import ExpToNat
from ..natural_parametrization import NaturalParametrization
from ..parameter import (IntegralRing, RealField, ScalarSupport, distribution_parameter,
                         negative_support)
from ..parametrization import SimpleDistribution

log_probability_floor = -50.0
log_probability_ceiling = -1e-7


@dataclass
class LogarithmicNP(NaturalParametrization['LogarithmicEP', JaxRealArray],
                    SimpleDistribution):
    """The natural parametrization of the logarithmic distribution.

    Args:
        log_probability: log(p).
    """
    log_probability: JaxRealArray = distribution_parameter(ScalarSupport(ring=negative_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.log_probability.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=IntegralRing(minimum=1))

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.log(-xp.log1p(-xp.exp(self.log_probability)))

    @override
    def to_exp(self) -> LogarithmicEP:
        xp = self.array_namespace()
        probability = xp.exp(self.log_probability)
        chi = xp.where(self.log_probability < log_probability_floor,
                       xp.asarray(1.0),
                       xp.where(self.log_probability > log_probability_ceiling,
                                xp.asarray(xp.inf),
                                probability / (xp.expm1(self.log_probability)
                                               * xp.log1p(-probability))))
        return LogarithmicEP(chi)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return -xp.log(x)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray
                              ) -> LogarithmicEP:
        return LogarithmicEP(x)


@dataclass
class LogarithmicEP(ExpToNat[LogarithmicNP],
                    ExpectationParametrization[LogarithmicNP],
                    SimpleDistribution):
    """The expectation parametrization of the logarithmic distribution.

    Args:
        chi: -(p / (1-p)) * log(1-p).
    """
    chi: JaxRealArray = distribution_parameter(ScalarSupport(ring=RealField(minimum=1.0)))

    @property
    @override
    def shape(self) -> Shape:
        return self.chi.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=IntegralRing(minimum=1))

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[LogarithmicNP]:
        return LogarithmicNP

    # The expected_carrier_measure is unknown.

    @override
    def to_nat(self) -> LogarithmicNP:
        xp = self.array_namespace()
        z: LogarithmicNP = super().to_nat()
        return LogarithmicNP(xp.where(self.chi < 1.0,
                                      xp.asarray(xp.nan),
                                      xp.where(self.chi == 1.0,
                                               xp.asarray(xp.inf),
                                               z.log_probability)))
