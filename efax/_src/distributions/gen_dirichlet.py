"""The generalized Dirichlet distribution.

It is based off:

T.-T. Wong 1998. Generalized Dirichlet distribution in Bayesian analysis. Applied Mathematics
and Computation, volume 97, pp165-181
"""
from __future__ import annotations

from typing import Any

from array_api_compat import get_namespace
from jax.scipy.special import betaln, digamma
from tjax import JaxRealArray, Shape, softplus
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.multidimensional import Multidimensional
from ..mixins.exp_to_nat.exp_to_nat import ExpToNat
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import RealField, VectorSupport, distribution_parameter, negative_support


@dataclass
class GeneralizedDirichletNP(HasEntropyNP['GeneralizedDirichletEP'],
                             NaturalParametrization['GeneralizedDirichletEP', JaxRealArray],
                             Multidimensional):
    alpha_minus_one: JaxRealArray = distribution_parameter(VectorSupport(
        ring=RealField(minimum=-1.0, generation_scale=3.0)))
    gamma: JaxRealArray = distribution_parameter(VectorSupport(
        ring=RealField(minimum=0.0, generation_scale=3.0)))

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
        xp = self.get_namespace()
        alpha, beta = self.alpha_beta()
        return xp.sum(betaln(alpha, beta), axis=-1)

    @override
    def to_exp(self) -> GeneralizedDirichletEP:
        # Given a log-normalizer y.
        # alpha_bar_direct = d y / d alpha = betaln'(alpha, beta) = digamma(alpha) - digamma(alpha +
        # beta)
        # beta_bar = d y / d beta = betaln'(alpha, beta) = digamma(beta) - digamma(alpha + beta)
        xp = self.get_namespace()
        alpha, beta = self.alpha_beta()
        digamma_sum = digamma(alpha + beta)
        alpha_bar_direct = digamma(alpha) - digamma_sum
        beta_bar = digamma(beta) - digamma_sum
        cs_beta_bar = xp.cumulative_sum(beta_bar, axis=-1, include_initial=True)
        gamma_bar = cs_beta_bar[..., 1:]
        alpha_bar_indirect = cs_beta_bar[..., :-1]
        alpha_bar = alpha_bar_direct + alpha_bar_indirect
        return GeneralizedDirichletEP(alpha_bar, gamma_bar)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> GeneralizedDirichletEP:
        xp = get_namespace(x)
        cs_x = xp.cumulative_sum(x, axis=-1)
        # cs_x[i] = sum_{j<=i} x[j]
        return GeneralizedDirichletEP(xp.log(x), xp.log1p(-cs_x))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.get_namespace(x)
        return xp.zeros(x.shape[:len(x.shape) - 1])

    @override
    def dimensions(self) -> int:
        return self.alpha_minus_one.shape[-1]

    def alpha_beta(self) -> tuple[JaxRealArray, JaxRealArray]:
        xp = self.get_namespace()
        alpha = self.alpha_minus_one + 1.0
        # cs_alpha[i] = sum_{j>=i} alpha[j]
        # cs_gamma[i] = sum_{j>=i} gamma[j]
        cs_alpha = xp.cumulative_sum(alpha[..., ::-1], axis=-1, include_initial=True)[..., :-1]
        cs_gamma = xp.cumulative_sum(self.gamma[..., ::-1], axis=-1)
        beta = (cs_gamma + cs_alpha + 1.0)[..., ::-1]
        return alpha, beta


@dataclass
class GeneralizedDirichletEP(HasEntropyEP[GeneralizedDirichletNP],
                             ExpToNat[GeneralizedDirichletNP],
                             ExpectationParametrization[GeneralizedDirichletNP],
                             Multidimensional):
    # E({log(x_i)}_i)
    mean_log_probability: JaxRealArray = distribution_parameter(VectorSupport(
        ring=negative_support))
    # E({log(1-∑_{j≤i} x_j)}_i)
    mean_log_cumulative_probability: JaxRealArray = distribution_parameter(VectorSupport(
        ring=negative_support))

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
        xp = self.get_namespace()
        return xp.zeros(self.shape)

    @override
    def dimensions(self) -> int:
        return self.mean_log_probability.shape[-1]
