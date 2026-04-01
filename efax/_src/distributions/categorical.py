from __future__ import annotations

from typing import Self, override

import array_api_extra as xpx
import jax.random as jr
import jax.scipy.special as jss
import numpy as np
from array_api_compat import array_namespace
from tjax import JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass

from efax._src.interfaces.conjugate_prior import HasGeneralizedConjugatePrior
from efax._src.interfaces.multidimensional import Multidimensional
from efax._src.interfaces.samplable import Samplable
from efax._src.mixins.has_entropy import HasEntropyEP, HasEntropyNP
from efax._src.natural_parametrization import NaturalParametrization
from efax._src.parameter import RealField, VectorSupport, boolean_ring, distribution_parameter

from .dirichlet import DirichletNP
from .gen_dirichlet import GeneralizedDirichletNP


@dataclass
class CategoricalNP(
    HasEntropyNP["CategoricalEP"],
    Samplable,
    NaturalParametrization["CategoricalEP", JaxRealArray],
    Multidimensional,
):
    """The natural parametrization of the categorical distribution.

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
        return VectorSupport(ring=boolean_ring)

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = array_namespace(self)
        max_q = xp.maximum(0.0, xp.max(self.log_odds, axis=-1))
        q_minus_max_q = self.log_odds - max_q[..., np.newaxis]
        log_scaled_a = xp.logaddexp(-max_q, jss.logsumexp(q_minus_max_q, axis=-1))
        return max_q + log_scaled_a

    @override
    def to_exp(self) -> CategoricalEP:
        xp = array_namespace(self)
        max_q = xp.maximum(0.0, xp.max(self.log_odds, axis=-1))
        q_minus_max_q = self.log_odds - max_q[..., np.newaxis]
        log_scaled_a = xp.logaddexp(-max_q, jss.logsumexp(q_minus_max_q, axis=-1))
        return CategoricalEP(xp.exp(q_minus_max_q - log_scaled_a[..., np.newaxis]))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = array_namespace(self, x)
        return xp.zeros(x.shape[:-1])

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray) -> CategoricalEP:
        return CategoricalEP(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        xp = array_namespace(self)
        if shape is not None:
            shape += self.shape
        logits = xp.concat((self.log_odds, xp.zeros((*self.shape, 1))), axis=-1)
        retval = xpx.one_hot(jr.categorical(key, logits, shape=shape), self.dimensions() + 1)  # type: ignore
        assert isinstance(retval, JaxArray)
        return retval[..., :-1]  # pyright: ignore

    @override
    def dimensions(self) -> int:
        return self.log_odds.shape[-1]

    def nat_to_probability(self) -> JaxRealArray:
        xp = array_namespace(self)
        max_q = xp.maximum(0.0, xp.max(self.log_odds, axis=-1))
        q_minus_max_q = self.log_odds - max_q[..., np.newaxis]
        log_scaled_a = xp.logaddexp(-max_q, jss.logsumexp(q_minus_max_q, axis=-1))
        p = xp.exp(q_minus_max_q - log_scaled_a[..., np.newaxis])
        final_p = 1.0 - xp.sum(p, axis=-1, keepdims=True)
        return xp.concat((p, final_p), axis=-1)

    def nat_to_surprisal(self) -> JaxRealArray:
        xp = array_namespace(self)
        total_p = self.nat_to_probability()
        return -xp.log(total_p)


@dataclass
class CategoricalEP(
    HasEntropyEP[CategoricalNP], Samplable, HasGeneralizedConjugatePrior, Multidimensional
):
    """The expectation parametrization of the categorical distribution.

    Args:
        probability: The probability vector with the final element omitted, i.e., [p_i]_{i in 1...n-1}.
    """

    probability: JaxRealArray = distribution_parameter(
        VectorSupport(ring=RealField(minimum=0.0, maximum=1.0))
    )

    @property
    @override
    def shape(self) -> Shape:
        return self.probability.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport(ring=boolean_ring)

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[CategoricalNP]:
        return CategoricalNP

    @override
    def to_nat(self) -> CategoricalNP:
        xp = array_namespace(self)
        p_k = 1.0 - xp.sum(self.probability, axis=-1, keepdims=True)
        return CategoricalNP(xp.log(self.probability / p_k))

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = array_namespace(self)
        return xp.zeros(self.shape)

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> DirichletNP:
        xp = array_namespace(self)
        reshaped_n = n[..., np.newaxis]
        final_p = 1.0 - xp.sum(self.probability, axis=-1, keepdims=True)
        return DirichletNP(reshaped_n * xp.concat((self.probability, final_p), axis=-1))

    @classmethod
    @override
    def from_conjugate_prior_distribution(
        cls,
        cp: NaturalParametrization,
    ) -> tuple[Self, JaxRealArray]:
        assert isinstance(cp, DirichletNP)
        a = cp.alpha_minus_one
        xp = array_namespace(a)
        n = xp.sum(a, axis=-1)
        probability = a[..., :-1] / n[..., xp.newaxis]
        return cls(probability), n

    @override
    def generalized_conjugate_prior_distribution(self, n: JaxRealArray) -> GeneralizedDirichletNP:
        xp = array_namespace(self)
        tail_n = xp.cumulative_sum(n[..., ::-1], axis=-1)[..., ::-1]
        residual_probability = 1.0 - xp.cumulative_sum(self.probability, axis=-1)
        alpha_minus_one = self.probability * tail_n
        gamma = n * residual_probability
        return GeneralizedDirichletNP(alpha_minus_one, gamma)

    @override
    def as_conjugate_prior_observation(self) -> JaxRealArray:
        return self.probability

    @override
    def dimensions(self) -> int:
        return self.probability.shape[-1]

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_nat().sample(key, shape)
