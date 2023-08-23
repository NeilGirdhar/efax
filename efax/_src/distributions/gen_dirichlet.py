"""The generalized Dirichlet distribution.

It is based off:

T.-T. Wong 1998. Generalized Dirichlet distribution in Bayesian analysis. Applied Mathematics
and Computation, volume 97, pp165-181
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax.nn import softplus
from jax.scipy.special import digamma
from tjax import JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.multidimensional import Multidimensional
from ..mixins.exp_to_nat import ExpToNat
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import VectorSupport, distribution_parameter
from ..tools import betaln

__all__ = ['GeneralizedDirichletNP', 'GeneralizedDirichletEP']


@dataclass
class GeneralizedDirichletNP(HasEntropyNP['GeneralizedDirichletEP'],
                             NaturalParametrization['GeneralizedDirichletEP', JaxRealArray],
                             Multidimensional):
    alpha_minus_one: JaxRealArray = distribution_parameter(VectorSupport())
    gamma: JaxRealArray = distribution_parameter(VectorSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.alpha_minus_one.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        alpha, beta = self.alpha_beta()
        return jnp.sum(betaln(alpha, beta), axis=-1)

    @override
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

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> GeneralizedDirichletEP:
        cs_x = jnp.cumsum(x, axis=-1)
        # cs_x[i] = sum_{j<=i} x[j]
        return GeneralizedDirichletEP(jnp.log(x), jnp.log(1.0 - cs_x))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape[:len(x.shape) - 1])

    @override
    def dimensions(self) -> int:
        return self.alpha_minus_one.shape[-1]

    def alpha_beta(self) -> tuple[JaxRealArray, JaxRealArray]:
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
class GeneralizedDirichletEP(HasEntropyEP[GeneralizedDirichletNP],
                             ExpToNat[GeneralizedDirichletNP, JaxRealArray],
                             ExpectationParametrization[GeneralizedDirichletNP],
                             Multidimensional):
    # E({log(x_i)}_i)
    mean_log_probability: JaxRealArray = distribution_parameter(VectorSupport())
    # E({log(1-∑_{j≤i} x_j)}_i)
    mean_log_cumulative_probability: JaxRealArray = distribution_parameter(VectorSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean_log_probability.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[GeneralizedDirichletNP]:
        return GeneralizedDirichletNP

    @override
    def search_to_natural(self, search_parameters: JaxRealArray) -> GeneralizedDirichletNP:
        # Run Newton's method on the whole real hyperspace.
        n = self.dimensions()
        positive_search_parameters = softplus(search_parameters)
        return GeneralizedDirichletNP(positive_search_parameters[..., :n] - 1.0,
                                      positive_search_parameters[..., n:])

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.shape)

    @override
    def initial_search_parameters(self) -> JaxRealArray:
        return jnp.zeros((*self.shape, self.dimensions() * 2))

    @override
    def search_gradient(self, search_parameters: JaxRealArray) -> JaxRealArray:
        return self._natural_gradient(self.search_to_natural(search_parameters)).flattened()

    @override
    def dimensions(self) -> int:
        return self.mean_log_probability.shape[-1]
