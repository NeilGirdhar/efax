from __future__ import annotations

import jax.random as jr
from array_api_compat import array_namespace
from tjax import JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import NaturalParametrization
from ..parameter import RealField, ScalarSupport
from .dirichlet_common import DirichletCommonEP, DirichletCommonNP


@dataclass
class BetaNP(DirichletCommonNP['BetaEP'],
             NaturalParametrization['BetaEP', JaxRealArray]
             ):
    """The natural parametrization of the beta distribution.

    The best way to interpret the parameters of the beta distribution are that an observation x in
    [0, 1] represents the Bernoulli probability that outcome 0 (out of {0, 1}) is realized.  In this
    way, the Beta class coincides with a special case of the Dirichlet class.

    Args:
        alpha_minus_one: The shape parameters alpha, beta in an array with final dimension 2.
    """
    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=RealField(minimum=0.0, maximum=1.0))

    @override
    def to_exp(self) -> BetaEP:
        return BetaEP(self._exp_helper())

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray
                              ) -> BetaEP:
        xp = array_namespace(x)
        return BetaEP(xp.stack([xp.log(x), xp.log1p(-x)], axis=-1))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = array_namespace(x)
        return xp.zeros(x.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        alpha = 1.0 + self.alpha_minus_one
        return jr.beta(key, alpha[..., 0], alpha[..., 1], shape)


@dataclass
class BetaEP(DirichletCommonEP[BetaNP],
             ExpectationParametrization[BetaNP]):
    """The expectation parametrization of the beta distribution.

    The best way to interpret the parameters of the beta distribution are that an observation x in
    [0, 1] represents the Bernoulli probability that outcome 0 (out of {0, 1}) is realized.  In this
    way, the Beta class coincides with a special case of the Dirichlet class.

    Args:
        mean_log_probability: An array with final dimension [E(log(x)), E(log(1-x))].
    """
    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=RealField(minimum=0.0, maximum=1.0))

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[BetaNP]:
        return BetaNP
