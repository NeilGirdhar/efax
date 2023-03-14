from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import KeyArray
from jax.scipy import special as jss
from tjax import BooleanArray, JaxRealArray, Shape
from tjax.dataclasses import dataclass

from ..conjugate_prior import HasConjugatePrior
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter
from ..samplable import Samplable
from .beta import BetaNP

__all__ = ['BernoulliNP', 'BernoulliEP']


@dataclass
class BernoulliNP(NaturalParametrization['BernoulliEP', JaxRealArray]):
    log_odds: JaxRealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.log_odds.shape

    def log_normalizer(self) -> JaxRealArray:
        return jnp.logaddexp(self.log_odds, 0.0)

    def to_exp(self) -> BernoulliEP:
        return BernoulliEP(jss.expit(self.log_odds))

    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape)

    def sufficient_statistics(self, x: JaxRealArray) -> BernoulliEP:
        return BernoulliEP(x)

    # New methods ----------------------------------------------------------------------------------
    def nat_to_probability(self) -> JaxRealArray:
        p = jss.expit(self.log_odds)
        final_p = 1.0 - p
        return jnp.stack([p, final_p], axis=-1)

    def nat_to_surprisal(self) -> JaxRealArray:
        total_p = self.nat_to_probability()
        return -jnp.log(total_p)


@dataclass
class BernoulliEP(HasConjugatePrior[BernoulliNP], Samplable):
    probability: JaxRealArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.probability.shape

    @classmethod
    def natural_parametrization_cls(cls) -> type[BernoulliNP]:
        return BernoulliNP

    def to_nat(self) -> BernoulliNP:
        return BernoulliNP(jss.logit(self.probability))

    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.shape)

    def sample(self, key: KeyArray, shape: Shape | None = None) -> BooleanArray:
        if shape is not None:
            shape += self.shape
        return jax.random.bernoulli(key, self.probability, shape)

    def conjugate_prior_distribution(self, n: JaxRealArray) -> BetaNP:
        reshaped_n = n[..., np.newaxis]
        return BetaNP(reshaped_n * jnp.stack([self.probability, (1.0 - self.probability)], axis=-1))

    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.probability
