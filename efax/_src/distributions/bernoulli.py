from __future__ import annotations

from typing import Any, Self

import jax.random as jr
import jax.scipy.special as jss
import numpy as np
from array_api_compat import array_namespace
from tjax import JaxArray, JaxBooleanArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.conjugate_prior import HasConjugatePrior
from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import RealField, ScalarSupport, boolean_ring, distribution_parameter
from ..parametrization import SimpleDistribution
from .beta import BetaNP


@dataclass
class BernoulliNP(HasEntropyNP['BernoulliEP'],
                  NaturalParametrization['BernoulliEP', JaxRealArray],
                  Samplable,
                  SimpleDistribution):
    """The natural parametrization of the Bernoulli distribution.

    Args:
        log_odds: log(p / (1-p)).
    """
    log_odds: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.log_odds.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=boolean_ring)

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.logaddexp(self.log_odds, xp.asarray(0.0))

    @override
    def to_exp(self) -> BernoulliEP:
        return BernoulliEP(jss.expit(self.log_odds))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return xp.zeros(x.shape)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray
                              ) -> BernoulliEP:
        return BernoulliEP(x)

    def nat_to_probability(self) -> JaxRealArray:
        xp = self.array_namespace()
        p = jss.expit(self.log_odds)
        final_p = 1.0 - p
        return xp.stack([p, final_p], axis=-1)

    def nat_to_surprisal(self) -> JaxRealArray:
        xp = self.array_namespace()
        total_p = self.nat_to_probability()
        return -xp.log(total_p)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)


@dataclass
class BernoulliEP(HasEntropyEP[BernoulliNP],
                  HasConjugatePrior,
                  Samplable):
    """The expectation parametrization of the Bernoulli distribution.

    Args:
        probability: p.
    """
    probability: JaxRealArray = distribution_parameter(ScalarSupport(ring=RealField(
        minimum=0.0, maximum=1.0)))

    @property
    @override
    def shape(self) -> Shape:
        return self.probability.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=boolean_ring)

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[BernoulliNP]:
        return BernoulliNP

    @override
    def to_nat(self) -> BernoulliNP:
        return BernoulliNP(jss.logit(self.probability))

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxBooleanArray:
        if shape is not None:
            shape += self.shape
        return jr.bernoulli(key, self.probability, shape)

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> BetaNP:
        xp = self.array_namespace()
        reshaped_n = n[..., np.newaxis]
        return BetaNP(reshaped_n * xp.stack([self.probability, (1.0 - self.probability)], axis=-1))

    @classmethod
    @override
    def from_conjugate_prior_distribution(cls, cp: NaturalParametrization[Any, Any]
                                          ) -> tuple[Self, JaxRealArray]:
        assert isinstance(cp, BetaNP)
        a = cp.alpha_minus_one
        xp = array_namespace(a)
        n = xp.sum(a, axis=-1)
        probability = (a[..., 0] / n)
        return (cls(probability), n)

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.probability
