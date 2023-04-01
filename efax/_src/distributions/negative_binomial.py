from __future__ import annotations

import jax.numpy as jnp
from jax.random import KeyArray, split
from tjax import JaxIntegralArray, JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.samplable import Samplable
from ..parameter import ScalarSupport, distribution_parameter
from .gamma import GammaNP
from .negative_binomial_common import NBCommonEP, NBCommonNP
from .poisson import PoissonEP

__all__ = ['NegativeBinomialNP', 'NegativeBinomialEP']


@dataclass
class NegativeBinomialNP(NBCommonNP['NegativeBinomialEP']):
    """The negative binomial distribution.

    Models the number of Bernoulli trials having probability p until r failures.
    """
    log_not_p: JaxRealArray = distribution_parameter(ScalarSupport())
    failures: JaxIntegralArray = distribution_parameter(ScalarSupport(), fixed=True)

    @override
    def to_exp(self) -> NegativeBinomialEP:
        return NegativeBinomialEP(self._mean(), self.failures)

    @override
    def sufficient_statistics(self, x: JaxRealArray) -> NegativeBinomialEP:
        return NegativeBinomialEP(x, jnp.broadcast_to(self.failures, x.shape))

    @override
    def _failures(self) -> JaxIntegralArray:
        return self.failures


@dataclass
class NegativeBinomialEP(NBCommonEP[NegativeBinomialNP], Samplable):
    mean: JaxRealArray = distribution_parameter(ScalarSupport())
    failures: JaxIntegralArray = distribution_parameter(ScalarSupport(), fixed=True)

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[NegativeBinomialNP]:
        return NegativeBinomialNP

    @override
    def to_nat(self) -> NegativeBinomialNP:
        return NegativeBinomialNP(self._log_not_p(), self.failures)

    @override
    def _failures(self) -> JaxIntegralArray:
        return self.failures

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        gamma_key, poisson_key = split(key)
        r = self._failures()
        negative_rate = -r / self.mean
        shape_minus_one = r - 1.0
        g = GammaNP(negative_rate, shape_minus_one)
        lambda_ = g.sample(gamma_key, shape)
        p = PoissonEP(lambda_)
        return p.sample(poisson_key)
