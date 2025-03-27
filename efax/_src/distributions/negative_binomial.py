from __future__ import annotations

import jax.random as jr
from array_api_compat import array_namespace
from tjax import JaxArray, JaxIntegralArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.samplable import Samplable
from ..natural_parametrization import NaturalParametrization
from ..parameter import (IntegralRing, ScalarSupport, distribution_parameter, negative_support,
                         positive_support)
from .gamma import GammaNP
from .negative_binomial_common import NBCommonEP, NBCommonNP
from .poisson import PoissonEP


@dataclass
class NegativeBinomialNP(Samplable,
                         NBCommonNP['NegativeBinomialEP'],
                         NaturalParametrization['NegativeBinomialEP', JaxRealArray]):
    """The natural parametrization of the negative binomial distribution.

    Models the number of Bernoulli trials having probability p until r failures.

    Args:
        log_not_p: log(1-p).
        failures: r.
    """
    log_not_p: JaxRealArray = distribution_parameter(ScalarSupport(ring=negative_support))
    failures: JaxIntegralArray = distribution_parameter(ScalarSupport(ring=IntegralRing(minimum=1)),
                                                        fixed=True)

    @override
    def to_exp(self) -> NegativeBinomialEP:
        return NegativeBinomialEP(self._mean(), self.failures)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray
                              ) -> NegativeBinomialEP:
        xp = array_namespace(x)
        return NegativeBinomialEP(x, xp.broadcast_to(fixed_parameters['failures'], x.shape))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)

    @override
    def _failures(self) -> JaxIntegralArray:
        return self.failures


@dataclass
class NegativeBinomialEP(Samplable,
                         NBCommonEP[NegativeBinomialNP],
                         ExpectationParametrization[NegativeBinomialNP]):
    """The expectation parametrization of the negative binomial distribution.

    Models the number of Bernoulli trials x having probability p until r failures.

    Args:
        mean: E(x).
        failures: r.
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))
    failures: JaxIntegralArray = distribution_parameter(ScalarSupport(ring=IntegralRing(minimum=1)),
                                                        fixed=True)

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
        gamma_key, poisson_key = jr.split(key)
        r = self._failures()
        negative_rate = -r / self.mean
        shape_minus_one = r - 1.0
        g = GammaNP(negative_rate, shape_minus_one)
        lambda_ = g.sample(gamma_key, shape)
        p = PoissonEP(lambda_)
        return p.sample(poisson_key)
