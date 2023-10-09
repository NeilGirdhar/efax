from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import one_hot
from jax.scipy import special as jss
from tjax import JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.conjugate_prior import HasGeneralizedConjugatePrior
from ..interfaces.multidimensional import Multidimensional
from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import VectorSupport, distribution_parameter
from .dirichlet import DirichletNP
from .gen_dirichlet import GeneralizedDirichletNP

__all__ = ['MultinomialNP', 'MultinomialEP']


@dataclass
class MultinomialNP(HasEntropyNP['MultinomialEP'],
                    Samplable,
                    NaturalParametrization['MultinomialEP', JaxRealArray], Multidimensional):
    """The natural parametrization of the multinomial distribution.

    Args:
        log_odds: [log(p_i / p_n)]_{i in 1...n-1}.
    """
    log_odds: JaxRealArray = distribution_parameter(VectorSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.log_odds.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        max_q = jnp.maximum(0.0, jnp.amax(self.log_odds, axis=-1))
        q_minus_max_q = self.log_odds - max_q[..., np.newaxis]
        log_scaled_a = jnp.logaddexp(-max_q, jss.logsumexp(q_minus_max_q, axis=-1))
        return max_q + log_scaled_a

    @override
    def to_exp(self) -> MultinomialEP:
        max_q = jnp.maximum(0.0, jnp.amax(self.log_odds, axis=-1))
        q_minus_max_q = self.log_odds - max_q[..., np.newaxis]
        log_scaled_a = jnp.logaddexp(-max_q, jss.logsumexp(q_minus_max_q, axis=-1))
        return MultinomialEP(jnp.exp(q_minus_max_q - log_scaled_a[..., np.newaxis]))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape[:-1])

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> MultinomialEP:
        return MultinomialEP(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        return one_hot(jax.random.categorical(key, self.log_odds, shape=shape),
                       self.dimensions())

    @override
    def dimensions(self) -> int:
        return self.log_odds.shape[-1]

    def nat_to_probability(self) -> JaxRealArray:
        max_q = jnp.maximum(0.0, jnp.amax(self.log_odds, axis=-1))
        q_minus_max_q = self.log_odds - max_q[..., np.newaxis]
        log_scaled_a = jnp.logaddexp(-max_q, jss.logsumexp(q_minus_max_q, axis=-1))
        p = jnp.exp(q_minus_max_q - log_scaled_a[..., np.newaxis])
        final_p = 1.0 - jnp.sum(p, axis=-1, keepdims=True)
        return jnp.append(p, final_p, axis=-1)

    def nat_to_surprisal(self) -> JaxRealArray:
        total_p = self.nat_to_probability()
        return -jnp.log(total_p)


@dataclass
class MultinomialEP(HasEntropyEP[MultinomialNP],
                    HasGeneralizedConjugatePrior,
                    ExpectationParametrization[MultinomialNP],
                    Multidimensional):
    """The expectation parametrization of the multinomial distribution.

    Args:
        log_odds: The probability vector with the final element omitted, i.e., [p_i]_{i in 1...n-1}.
    """
    probability: JaxRealArray = distribution_parameter(VectorSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.probability.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[MultinomialNP]:
        return MultinomialNP

    @override
    def to_nat(self) -> MultinomialNP:
        p_k = 1.0 - jnp.sum(self.probability, axis=-1, keepdims=True)
        return MultinomialNP(jnp.log(self.probability / p_k))

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.shape)

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> DirichletNP:
        reshaped_n = n[..., np.newaxis]
        final_p = 1.0 - jnp.sum(self.probability, axis=-1, keepdims=True)
        return DirichletNP(reshaped_n * jnp.append(self.probability, final_p, axis=-1))

    @override
    def generalized_conjugate_prior_distribution(self, n: JaxRealArray) -> GeneralizedDirichletNP:
        final_p = 1.0 - jnp.sum(self.probability, axis=-1, keepdims=True)
        all_p = jnp.append(self.probability, final_p, axis=-1)
        alpha = n * all_p
        beta = n * (1.0 - all_p)
        alpha_roll = jnp.roll(alpha, -1, axis=-1).at[..., -1].set(0.0)
        gamma = -jnp.diff(beta, append=1.0) - alpha_roll
        return GeneralizedDirichletNP(alpha - 1.0, gamma)

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.probability

    @override
    def dimensions(self) -> int:
        return self.probability.shape[-1]
