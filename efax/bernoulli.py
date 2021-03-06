from __future__ import annotations

import numpy as np
from jax import numpy as jnp
from jax.scipy import special as jss
from tjax import RealArray, Shape, dataclass

from .beta import BetaNP
from .conjugate_prior import HasConjugatePrior
from .natural_parametrization import NaturalParametrization
from .parameter import distribution_parameter

__all__ = ['BernoulliNP', 'BernoulliEP']


@dataclass
class BernoulliNP(NaturalParametrization['BernoulliEP']):
    log_odds: RealArray = distribution_parameter(axes=0)

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.log_odds.shape

    def log_normalizer(self) -> RealArray:
        return jnp.logaddexp(self.log_odds, 0.0)

    def to_exp(self) -> BernoulliEP:
        return BernoulliEP(jss.expit(self.log_odds))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape)

    def sufficient_statistics(self, x: RealArray) -> BernoulliEP:
        return BernoulliEP(x)

    # New methods ----------------------------------------------------------------------------------
    def nat_to_probability(self) -> RealArray:
        p = jss.expit(self.log_odds)
        final_p = 1.0 - p
        return jnp.stack([p, final_p], axis=-1)

    def nat_to_surprisal(self) -> RealArray:
        total_p = self.nat_to_probability()
        return -jnp.log(total_p)


@dataclass
class BernoulliEP(HasConjugatePrior[BernoulliNP]):
    probability: RealArray = distribution_parameter(axes=0)

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.probability.shape

    def to_nat(self) -> BernoulliNP:
        return BernoulliNP(jss.logit(self.probability))

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape())

    # Overridden methods ---------------------------------------------------------------------------
    def conjugate_prior_distribution(self, n: RealArray) -> BetaNP:
        reshaped_n = n[..., np.newaxis]
        return BetaNP(reshaped_n * jnp.stack([self.probability, (1.0 - self.probability)], axis=-1))

    def conjugate_prior_observation(self) -> RealArray:
        return self.probability
