from __future__ import annotations

from typing import Any, Self

import jax
from tjax import JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.conjugate_prior import HasConjugatePrior
from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter, negative_support, positive_support
from .gamma import GammaNP


@dataclass
class ExponentialNP(HasEntropyNP['ExponentialEP'],
                    Samplable,
                    NaturalParametrization['ExponentialEP', JaxRealArray]):
    """The natural parametrization of the exponential distribution.

    Args:
        negative_rate: -lambda where lambda is the rate.
    """
    negative_rate: JaxRealArray = distribution_parameter(ScalarSupport(ring=negative_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.negative_rate.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.get_namespace()
        return -xp.log(-self.negative_rate)

    @override
    def to_exp(self) -> ExponentialEP:
        return ExponentialEP(-1.0 / self.negative_rate)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.get_namespace(x)
        return xp.zeros(x.shape)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> ExponentialEP:
        return ExponentialEP(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        return -jax.random.exponential(key, shape) / self.negative_rate


@dataclass
class ExponentialEP(HasEntropyEP[ExponentialNP],
                    HasConjugatePrior,
                    Samplable,
                    ExpectationParametrization[ExponentialNP]):
    """The expectation parametrization of the exponential distribution.

    Args:
        mean: E(x) = 1/lambda where lambda is the rate.
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[ExponentialNP]:
        return ExponentialNP

    @override
    def to_nat(self) -> ExponentialNP:
        return ExponentialNP(-1.0 / self.mean)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.get_namespace()
        return xp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        return jax.random.exponential(key, shape) * self.mean

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> GammaNP:
        return GammaNP(-n / self.mean, n)

    @classmethod
    @override
    def from_conjugate_prior_distribution(cls, cp: NaturalParametrization[Any, Any]
                                          ) -> tuple[Self, JaxRealArray]:
        assert isinstance(cp, GammaNP)
        n = cp.shape_minus_one
        return (cls(-n / cp.negative_rate), n)

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.mean
