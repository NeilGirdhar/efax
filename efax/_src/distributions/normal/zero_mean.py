from __future__ import annotations
import jax.random as jr

import math
from typing import Any, Self

from tjax import JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...expectation_parametrization import ExpectationParametrization
from ...interfaces.conjugate_prior import HasConjugatePrior
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import ScalarSupport, distribution_parameter, negative_support, positive_support
from .normal import NormalNP


@dataclass
class ZeroMeanNormalNP(HasEntropyNP['ZeroMeanNormalEP'],
                       NaturalParametrization['ZeroMeanNormalEP', JaxRealArray],
                       Samplable):
    """The natural parametrization of the normal distribution with zero mean.

    This is a curved exponential family.
    """
    negative_half_precision: JaxRealArray = distribution_parameter(ScalarSupport(
        ring=negative_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.negative_half_precision.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.array_namespace()
        return 0.5 * xp.log(-math.pi / self.negative_half_precision)

    @override
    def to_exp(self) -> ZeroMeanNormalEP:
        return ZeroMeanNormalEP(-0.5 / self.negative_half_precision)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        # The second moment of a delta distribution at x.
        xp = self.array_namespace(x)
        return xp.zeros_like(x)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray,
                              **fixed_parameters: JaxArray
                              ) -> ZeroMeanNormalEP:
        return ZeroMeanNormalEP(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)


@dataclass
class ZeroMeanNormalEP(HasEntropyEP[ZeroMeanNormalNP],
                       HasConjugatePrior,
                       Samplable,
                       ExpectationParametrization[ZeroMeanNormalNP]):
    """The expectation parametrization of the normal distribution with zero mean.

    This is a curved exponential family.
    """
    variance: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.variance.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[ZeroMeanNormalNP]:
        return ZeroMeanNormalNP

    @override
    def to_nat(self) -> ZeroMeanNormalNP:
        return ZeroMeanNormalNP(-0.5 / self.variance)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        # The second moment of a normal distribution with the given mean.
        xp = self.array_namespace()
        return xp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        xp = self.array_namespace()
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        deviation = xp.sqrt(self.variance)
        return jr.normal(key, shape) * deviation

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> NormalNP:
        negative_half_precision = -0.5 * n
        return NormalNP(n * self.mean, negative_half_precision)

    @classmethod
    @override
    def from_conjugate_prior_distribution(cls, cp: NaturalParametrization[Any, Any]
                                          ) -> tuple[Self, JaxRealArray]:
        assert isinstance(cp, NormalNP)
        n = -2.0 * cp.negative_half_precision
        mean = cp.mean_times_precision / n
        return (cls(mean), n)

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.mean
