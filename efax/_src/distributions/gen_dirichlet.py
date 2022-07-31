"""
The generalized Dirichlet distribution is based off:

    T.-T. Wong 1998. Generalized Dirichlet distribution in Bayesian analysis. Applied Mathematics
    and Computation, volume 97, pp165-181
"""
from __future__ import annotations

import jax.numpy as jnp
from jax.nn import softplus
from jax.scipy.special import digamma
from tjax import RealArray, Shape
from tjax.dataclasses import dataclass

from ..exp_to_nat import ExpToNat
from ..multidimensional import Multidimensional
from ..natural_parametrization import NaturalParametrization
from ..parameter import VectorSupport, distribution_parameter
from ..tools import betaln

__all__ = ['GeneralizedDirichletNP', 'GeneralizedDirichletEP']


@dataclass
class GeneralizedDirichletNP(NaturalParametrization['GeneralizedDirichletEP', RealArray],
                             Multidimensional):
    alpha_minus_one: RealArray = distribution_parameter(VectorSupport())
    gamma: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.alpha_minus_one.shape[:-1]

    def log_normalizer(self) -> RealArray:
        alpha, beta = self.alpha_beta()
        return jnp.sum(betaln(alpha, beta), axis=-1)

    def to_exp(self) -> GeneralizedDirichletEP:
        # Given a log-normalizer y.
        # alpha_bar_direct = d y / d alpha = betaln'(alpha, beta) = digamma(alpha) - digamma(alpha +
        # beta)
        # beta_bar = d y / d beta = betaln'(alpha, beta) = digamma(beta) - digamma(alpha + beta)
        alpha, beta = self.alpha_beta()
        digamma_sum = digamma(alpha + beta)
        alpha_bar_direct = digamma(alpha) - digamma_sum
        beta_bar = digamma(beta) - digamma_sum
        gamma_bar = jnp.cumsum(beta_bar, axis=-1)
        alpha_bar_indirect = jnp.roll(gamma_bar, 1, axis=-1).at[..., 0].set(0.0)
        alpha_bar = alpha_bar_direct + alpha_bar_indirect
        return GeneralizedDirichletEP(alpha_bar, gamma_bar)

    def sufficient_statistics(self, x: RealArray) -> GeneralizedDirichletEP:
        cs_x = jnp.cumsum(x, axis=-1)
        # cs_x[i] = sum_{j<=i} x[j]
        return GeneralizedDirichletEP(jnp.log(x), jnp.log(1.0 - cs_x))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape[:len(x.shape) - 1])

    def dimensions(self) -> int:
        return self.alpha_minus_one.shape[-1]

    # New methods ----------------------------------------------------------------------------------
    def alpha_beta(self) -> tuple[RealArray, RealArray]:
        alpha = self.alpha_minus_one + 1.0
        # cs_alpha[i] = sum_{j>=i} alpha[j]
        # cs_gamma[i] = sum_{j>=i} gamma[j]
        cs_alpha = jnp.cumsum(alpha[..., ::-1], axis=-1)[..., ::-1]
        cs_gamma = jnp.cumsum(self.gamma[..., ::-1], axis=-1)[..., ::-1]
        # roll_cs_alpha[i] = sum_{j>i} alpha[j]
        roll_cs_alpha = jnp.roll(cs_alpha, -1, axis=-1)
        roll_cs_alpha = roll_cs_alpha.at[..., -1].set(0.0)
        beta = cs_gamma + roll_cs_alpha + 1.0
        return alpha, beta


@dataclass
class GeneralizedDirichletEP(ExpToNat[GeneralizedDirichletNP, RealArray], Multidimensional):
    # E({log(x_i)}_i)
    mean_log_probability: RealArray = distribution_parameter(VectorSupport())
    # E({log(1−∑_{j≤i} x_j)}_i)
    mean_log_cumulative_probability: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean_log_probability.shape[:-1]

    @classmethod
    def natural_parametrization_cls(cls) -> type[GeneralizedDirichletNP]:
        return GeneralizedDirichletNP

    def search_to_natural(self, search_parameters: RealArray) -> GeneralizedDirichletNP:
        # Run Newton's method on the whole real hyperspace.
        n = self.dimensions()
        positive_search_parameters = softplus(search_parameters)
        return GeneralizedDirichletNP(positive_search_parameters[..., :n] - 1.0,
                                      positive_search_parameters[..., n:])

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape)

    def initial_search_parameters(self) -> RealArray:
        return jnp.zeros(self.shape + (self.dimensions() * 2,))

    def search_gradient(self, search_parameters: RealArray) -> RealArray:
        return self._natural_gradient(self.search_to_natural(search_parameters)).flattened()

    def dimensions(self) -> int:
        return self.mean_log_probability.shape[-1]
