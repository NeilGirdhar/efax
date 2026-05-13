"""The generalized Dirichlet distribution.

It is based off:

T.-T. Wong 1998. Generalized Dirichlet distribution in Bayesian analysis. Applied Mathematics
and Computation, volume 97, pp165-181
"""

from __future__ import annotations

from typing import override

import jax.scipy.special as jss
from array_api_compat import array_namespace
from tjax import JaxArray, JaxRealArray, Shape, complex_betaln, inverse_softplus, jit, softplus
from tjax.dataclasses import dataclass

from efax._src.interfaces.multidimensional import Multidimensional
from efax._src.mixins.exp_to_nat.exp_to_nat import ExpToNat
from efax._src.mixins.has_entropy import HasEntropyEP, HasEntropyNP
from efax._src.natural_parametrization import NaturalParametrization
from efax._src.parameter import (
    RealField,
    SubsimplexSupport,
    VectorSupport,
    distribution_parameter,
    negative_support,
)


@dataclass
class GeneralizedDirichletNP(
    HasEntropyNP["GeneralizedDirichletEP"],
    NaturalParametrization["GeneralizedDirichletEP", JaxRealArray],
    Multidimensional,
):
    alpha_minus_one: JaxRealArray = distribution_parameter(
        VectorSupport(ring=RealField(minimum=-1.0, generation_scale=3.0))
    )
    gamma: JaxRealArray = distribution_parameter(
        VectorSupport(ring=RealField(minimum=0.0, generation_scale=3.0))
    )

    @property
    @override
    def shape(self) -> Shape:
        return self.alpha_minus_one.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> SubsimplexSupport:
        return SubsimplexSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = array_namespace(self)
        alpha, beta = self.alpha_beta()
        return xp.sum(complex_betaln(alpha, beta), axis=-1)

    @override
    def to_exp(self) -> GeneralizedDirichletEP:
        # Given a log-normalizer y.
        # alpha_bar_direct = d y / d alpha = jss.betaln'(alpha, beta)
        # = jss.digamma(alpha) - jss.digamma(alpha + beta)
        # beta_bar = d y / d beta = jss.betaln'(alpha, beta)
        # = jss.digamma(beta) - jss.digamma(alpha + beta)
        xp = array_namespace(self)
        alpha, beta = self.alpha_beta()
        digamma_sum = jss.digamma(alpha + beta)
        alpha_bar_direct = jss.digamma(alpha) - digamma_sum
        beta_bar = jss.digamma(beta) - digamma_sum
        cs_beta_bar = xp.cumulative_sum(beta_bar, axis=-1, include_initial=True)
        gamma_bar = cs_beta_bar[..., 1:]
        alpha_bar_indirect = cs_beta_bar[..., :-1]
        alpha_bar = alpha_bar_direct + alpha_bar_indirect
        return GeneralizedDirichletEP(alpha_bar, gamma_bar)

    @override
    @classmethod
    def sufficient_statistics(
        cls, x: JaxRealArray, **fixed_parameters: JaxArray
    ) -> GeneralizedDirichletEP:
        xp = array_namespace(x)
        cs_x = xp.cumulative_sum(x, axis=-1)
        # cs_x[i] = sum_{j<=i} x[j]
        return GeneralizedDirichletEP(xp.log(x), xp.log1p(-cs_x))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = array_namespace(self, x)
        return xp.zeros(x.shape[: len(x.shape) - 1])

    @override
    def dimensions(self) -> int:
        return self.alpha_minus_one.shape[-1]

    def alpha_beta(self) -> tuple[JaxRealArray, JaxRealArray]:
        xp = array_namespace(self)
        alpha = self.alpha_minus_one + 1.0
        # cs_alpha[i] = sum_{j>=i} alpha[j]
        # cs_gamma[i] = sum_{j>=i} gamma[j]
        cs_alpha = xp.cumulative_sum(alpha[..., ::-1], axis=-1, include_initial=True)[..., :-1]
        cs_gamma = xp.cumulative_sum(self.gamma[..., ::-1], axis=-1)
        beta = (cs_gamma + cs_alpha + 1.0)[..., ::-1]
        return alpha, beta


@dataclass
class GeneralizedDirichletEP(
    HasEntropyEP[GeneralizedDirichletNP], ExpToNat[GeneralizedDirichletNP], Multidimensional
):
    # E({log(x_i)}_i)
    mean_log_probability: JaxRealArray = distribution_parameter(
        VectorSupport(ring=negative_support)
    )
    # E({log(1-∑_{j≤i} x_j)}_i)
    mean_log_cumulative_probability: JaxRealArray = distribution_parameter(
        VectorSupport(ring=negative_support)
    )

    @property
    @override
    def shape(self) -> Shape:
        return self.mean_log_probability.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> SubsimplexSupport:
        return SubsimplexSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[GeneralizedDirichletNP]:
        return GeneralizedDirichletNP

    @jit
    @override
    def to_nat(self) -> GeneralizedDirichletNP:
        xp = array_namespace(self)
        zero = xp.zeros_like(self.mean_log_cumulative_probability[..., :1])
        beta_bar = xp.concat(
            [
                self.mean_log_cumulative_probability[..., :1],
                xp.diff(self.mean_log_cumulative_probability, axis=-1),
            ],
            axis=-1,
        )
        alpha_bar_direct = self.mean_log_probability - xp.concat(
            [zero, self.mean_log_cumulative_probability[..., :-1]],
            axis=-1,
        )
        alpha, beta = _beta_parameters_from_expected_logs(alpha_bar_direct, beta_bar)
        alpha_roll = xp.concat([alpha[..., 1:], zero], axis=-1)
        gamma = -xp.diff(beta, axis=-1, append=xp.ones_like(zero)) - alpha_roll
        return GeneralizedDirichletNP(alpha - 1.0, gamma)

    @override
    def initial_search_parameters(self) -> JaxRealArray:
        xp = array_namespace(self)
        beta_bar = xp.concat(
            [
                self.mean_log_cumulative_probability[..., :1],
                xp.diff(self.mean_log_cumulative_probability, axis=-1),
            ],
            axis=-1,
        )
        zero = xp.zeros_like(self.mean_log_cumulative_probability[..., :1])
        alpha_bar_direct = self.mean_log_probability - xp.concat(
            [zero, self.mean_log_cumulative_probability[..., :-1]],
            axis=-1,
        )
        alpha, beta = _beta_parameters_from_expected_logs(alpha_bar_direct, beta_bar)
        alpha_roll = xp.concat([alpha[..., 1:], zero], axis=-1)
        gamma = -xp.diff(beta, axis=-1, append=xp.ones_like(zero)) - alpha_roll
        gamma = xp.maximum(gamma, 1e-3)
        return xp.concat([inverse_softplus(alpha), inverse_softplus(gamma)], axis=-1)

    @override
    def search_to_natural(self, search_parameters: JaxRealArray) -> GeneralizedDirichletNP:
        # Run Newton's method on the whole real hyperspace.
        n = self.dimensions()
        positive_search_parameters = softplus(search_parameters)
        return GeneralizedDirichletNP(
            positive_search_parameters[..., :n] - 1.0, positive_search_parameters[..., n:]
        )

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = array_namespace(self)
        return xp.zeros(self.shape)

    @override
    def dimensions(self) -> int:
        return self.mean_log_probability.shape[-1]


inverse_digamma_switch = -2.22


def _inverse_digamma(y: JaxRealArray) -> JaxRealArray:
    xp = array_namespace(y)
    x = xp.where(
        y >= inverse_digamma_switch,
        xp.exp(y) + 0.5,
        -1.0 / (y - jss.digamma(1.0)),
    )
    for _ in range(8):
        x -= (jss.digamma(x) - y) / jss.polygamma(1, x)
    return x


def _beta_parameters_from_expected_logs(
    mean_log_probability: JaxRealArray,
    mean_log_complement: JaxRealArray,
) -> tuple[JaxRealArray, JaxRealArray]:
    xp = array_namespace(mean_log_probability, mean_log_complement)
    mean = xp.exp(mean_log_probability)
    complement = xp.exp(mean_log_complement)
    simplex_mean = mean / (mean + complement)
    concentration = xp.maximum(2.0, 1.0 / xp.maximum(simplex_mean * (1.0 - simplex_mean), 1e-3))
    for _ in range(20):
        psi_sum = jss.digamma(concentration)
        alpha = _inverse_digamma(mean_log_probability + psi_sum)
        beta = _inverse_digamma(mean_log_complement + psi_sum)
        concentration = xp.maximum(alpha + beta, 1e-6)
    alpha = xp.maximum(alpha, 1e-6)
    beta = xp.maximum(beta, 1e-6)
    log_alpha = xp.log(alpha)
    log_beta = xp.log(beta)
    for _ in range(20):
        alpha = xp.exp(log_alpha)
        beta = xp.exp(log_beta)
        delta_log_alpha, delta_log_beta = _beta_expected_log_newton_step(
            alpha, beta, mean_log_probability, mean_log_complement
        )
        log_alpha -= delta_log_alpha
        log_beta -= delta_log_beta
    alpha = xp.exp(log_alpha)
    beta = xp.exp(log_beta)
    return alpha, beta


def _beta_expected_log_newton_step(
    alpha: JaxRealArray,
    beta: JaxRealArray,
    mean_log_probability: JaxRealArray,
    mean_log_complement: JaxRealArray,
) -> tuple[JaxRealArray, JaxRealArray]:
    trigamma_sum = jss.polygamma(1, alpha + beta)
    f1 = jss.digamma(alpha) - jss.digamma(alpha + beta) - mean_log_probability
    f2 = jss.digamma(beta) - jss.digamma(alpha + beta) - mean_log_complement
    j11 = alpha * (jss.polygamma(1, alpha) - trigamma_sum)
    j12 = -beta * trigamma_sum
    j21 = -alpha * trigamma_sum
    j22 = beta * (jss.polygamma(1, beta) - trigamma_sum)
    determinant = j11 * j22 - j12 * j21
    delta_log_alpha = (j22 * f1 - j12 * f2) / determinant
    delta_log_beta = (-j21 * f1 + j11 * f2) / determinant
    return delta_log_alpha, delta_log_beta
