from __future__ import annotations

from typing import Iterable

import numpy as np
from jax import numpy as jnp
from jax.scipy import special as jss
from tjax import RealArray, Shape, dataclass

from .conjugate_prior import HasConjugatePrior
from .dirichlet import DirichletNP
from .exponential_family import NaturalParametrization

__all__ = ['MultinomialNP', 'MultinomialEP']


@dataclass
class MultinomialNP(NaturalParametrization['MultinomialEP']):

    log_odds: RealArray

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.log_odds.shape[:-1]

    def log_normalizer(self) -> RealArray:
        max_q = jnp.maximum(0.0, jnp.amax(self.log_odds, axis=-1))
        q_minus_max_q = self.log_odds - max_q[..., np.newaxis]
        log_scaled_A = jnp.logaddexp(-max_q, jss.logsumexp(q_minus_max_q, axis=-1))
        return max_q + log_scaled_A

    def to_exp(self) -> MultinomialEP:
        max_q = jnp.maximum(0.0, jnp.amax(self.log_odds, axis=-1))
        q_minus_max_q = self.log_odds - max_q[..., np.newaxis]
        log_scaled_A = jnp.logaddexp(-max_q, jss.logsumexp(q_minus_max_q, axis=-1))
        return MultinomialEP(jnp.exp(q_minus_max_q - log_scaled_A[..., np.newaxis]))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape[:-1])

    def sufficient_statistics(self, x: RealArray) -> MultinomialEP:
        return MultinomialEP(x)

    @classmethod
    def field_axes(cls) -> Iterable[int]:
        yield 1

    # New methods ----------------------------------------------------------------------------------
    def nat_to_probability(self) -> RealArray:
        max_q = jnp.maximum(0.0, jnp.amax(self.log_odds, axis=-1))
        q_minus_max_q = self.log_odds - max_q[..., np.newaxis]
        log_scaled_A = jnp.logaddexp(-max_q, jss.logsumexp(q_minus_max_q, axis=-1))
        p = jnp.exp(q_minus_max_q - log_scaled_A[..., np.newaxis])
        final_p = 1.0 - jnp.sum(p, axis=-1, keepdims=True)
        return jnp.append(p, final_p, axis=-1)

    def nat_to_surprisal(self) -> RealArray:
        total_p = self.nat_to_probability()
        return -jnp.log(total_p)


@dataclass
class MultinomialEP(HasConjugatePrior[MultinomialNP]):

    probability: RealArray

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.probability.shape[:-1]

    def to_nat(self) -> MultinomialNP:
        p_k = 1.0 - jnp.sum(self.probability, axis=-1, keepdims=True)
        return MultinomialNP(jnp.log(self.probability / p_k))

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape())

    # Overridden methods ---------------------------------------------------------------------------
    def conjugate_prior_distribution(self, n: RealArray) -> DirichletNP:
        reshaped_n = n[..., np.newaxis]
        final_p = 1.0 - jnp.sum(self.probability, axis=-1, keepdims=True)
        return DirichletNP(reshaped_n * jnp.append(self.probability, final_p, axis=-1))

    def conjugate_prior_observation(self) -> RealArray:
        return self.probability
