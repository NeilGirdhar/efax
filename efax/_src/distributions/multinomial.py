from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import one_hot
from jax.random import KeyArray
from jax.scipy import special as jss
from tjax import RealArray, Shape
from tjax.dataclasses import dataclass

from ..conjugate_prior import HasGeneralizedConjugatePrior
from ..multidimensional import Multidimensional
from ..natural_parametrization import NaturalParametrization
from ..parameter import VectorSupport, distribution_parameter
from ..samplable import Samplable
from .dirichlet import DirichletNP
from .gen_dirichlet import GeneralizedDirichletNP

__all__ = ['MultinomialNP', 'MultinomialEP']


@dataclass
class MultinomialNP(NaturalParametrization['MultinomialEP', RealArray], Multidimensional,
                    Samplable):
    log_odds: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
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

    def sample(self, rng: KeyArray, shape: Shape | None = None) -> RealArray:
        if shape is not None:
            shape += self.shape
        return one_hot(jax.random.categorical(rng, self.log_odds, shape=shape),
                       self.dimensions())

    def dimensions(self) -> int:
        return self.log_odds.shape[-1]

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
class MultinomialEP(HasGeneralizedConjugatePrior[MultinomialNP], Multidimensional):
    probability: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.probability.shape[:-1]

    @classmethod
    def natural_parametrization_cls(cls) -> type[MultinomialNP]:
        return MultinomialNP

    def to_nat(self) -> MultinomialNP:
        p_k = 1.0 - jnp.sum(self.probability, axis=-1, keepdims=True)
        return MultinomialNP(jnp.log(self.probability / p_k))

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape)

    def conjugate_prior_distribution(self, n: RealArray) -> DirichletNP:
        reshaped_n = n[..., np.newaxis]
        final_p = 1.0 - jnp.sum(self.probability, axis=-1, keepdims=True)
        return DirichletNP(reshaped_n * jnp.append(self.probability, final_p, axis=-1))

    def generalized_conjugate_prior_distribution(self, n: RealArray) -> GeneralizedDirichletNP:
        final_p = 1.0 - jnp.sum(self.probability, axis=-1, keepdims=True)
        all_p = jnp.append(self.probability, final_p, axis=-1)
        alpha = n * all_p
        beta = n * (1.0 - all_p)
        alpha_roll = jnp.roll(alpha, -1, axis=-1).at[..., -1].set(0.0)
        gamma = -jnp.diff(beta, append=1.0) - alpha_roll
        return GeneralizedDirichletNP(alpha - 1.0, gamma)

    def conjugate_prior_observation(self) -> RealArray:
        return self.probability

    def dimensions(self) -> int:
        return self.probability.shape[-1]
