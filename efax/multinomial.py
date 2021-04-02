from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import one_hot
from jax.scipy import special as jss
from tjax import Generator, RealArray, Shape, dataclass

from .conjugate_prior import HasConjugatePrior
from .dirichlet import DirichletNP
from .natural_parametrization import NaturalParametrization
from .parameter import VectorSupport, distribution_parameter
from .samplable import Samplable

__all__ = ['MultinomialNP', 'MultinomialEP']


@dataclass
class MultinomialNP(NaturalParametrization['MultinomialEP'], Samplable):
    log_odds: RealArray = distribution_parameter(VectorSupport())

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

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> RealArray:
        if shape is not None:
            shape += self.shape()
        return one_hot(jax.random.categorical(rng.key, self.log_odds, shape=shape),
                       self.log_odds.shape[-1])

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.log_odds.shape[-1]

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
    probability: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.probability.shape[:-1]

    def to_nat(self) -> MultinomialNP:
        p_k = 1.0 - jnp.sum(self.probability, axis=-1, keepdims=True)
        return MultinomialNP(jnp.log(self.probability / p_k))

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape())

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.probability.shape[-1]

    # Overridden methods ---------------------------------------------------------------------------
    def conjugate_prior_distribution(self, n: RealArray) -> DirichletNP:
        reshaped_n = n[..., np.newaxis]
        final_p = 1.0 - jnp.sum(self.probability, axis=-1, keepdims=True)
        return DirichletNP(reshaped_n * jnp.append(self.probability, final_p, axis=-1))

    def conjugate_prior_observation(self) -> RealArray:
        return self.probability
