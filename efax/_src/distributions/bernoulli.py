from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import special as jss
from tjax import JaxBooleanArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.conjugate_prior import HasConjugatePrior
from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, boolean_ring, distribution_parameter
from .beta import BetaNP

__all__ = ['BernoulliNP', 'BernoulliEP']


@dataclass
class BernoulliNP(HasEntropyNP['BernoulliEP'],
                  NaturalParametrization['BernoulliEP', JaxRealArray]):
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
        return jnp.logaddexp(self.log_odds, 0.0)

    @override
    def to_exp(self) -> BernoulliEP:
        return BernoulliEP(jss.expit(self.log_odds))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> BernoulliEP:
        return BernoulliEP(x)

    def nat_to_probability(self) -> JaxRealArray:
        p = jss.expit(self.log_odds)
        final_p = 1.0 - p
        return jnp.stack([p, final_p], axis=-1)

    def nat_to_surprisal(self) -> JaxRealArray:
        total_p = self.nat_to_probability()
        return -jnp.log(total_p)


@dataclass
class BernoulliEP(HasEntropyEP[BernoulliNP],
                  HasConjugatePrior,
                  Samplable,
                  ExpectationParametrization[BernoulliNP]):
    """The expectation parametrization of the Bernoulli distribution.

    Args:
        probability: p.
    """
    probability: JaxRealArray = distribution_parameter(ScalarSupport())

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
        return jnp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxBooleanArray:
        if shape is not None:
            shape += self.shape
        return jax.random.bernoulli(key, self.probability, shape)

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> BetaNP:
        reshaped_n = n[..., np.newaxis]
        return BetaNP(reshaped_n * jnp.stack([self.probability, (1.0 - self.probability)], axis=-1))

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.probability
