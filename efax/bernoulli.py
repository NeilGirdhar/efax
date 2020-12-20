from __future__ import annotations

import numpy as np
from jax import numpy as jnp
from jax.scipy import special as jss
from tjax import RealArray, dataclass

from .beta import BetaNP
from .multinomial import MultinomialEP, MultinomialNP

__all__ = ['BernoulliNP', 'BernoulliEP']


@dataclass
class BernoulliNP(MultinomialNP):

    # Overridden methods ---------------------------------------------------------------------------
    def log_normalizer(self) -> RealArray:
        q = self.log_odds[..., 0]
        return jnp.logaddexp(q, 0.0)

    def to_exp(self) -> BernoulliEP:
        return BernoulliEP(jss.expit(self.log_odds))

    def sufficient_statistics(self, x: RealArray) -> BernoulliEP:
        ss = super().sufficient_statistics(x)
        return BernoulliEP(ss.probability)


@dataclass
class BernoulliEP(MultinomialEP):

    # Implemented methods --------------------------------------------------------------------------
    def conjugate_prior_distribution(self, n: RealArray) -> BetaNP:
        reshaped_n = n[..., np.newaxis]
        return BetaNP(reshaped_n * jnp.append(self.probability, (1.0 - self.probability), axis=-1))

    # Overridden methods ---------------------------------------------------------------------------
    def to_nat(self) -> BernoulliNP:
        return BernoulliNP(jss.logit(self.probability))
