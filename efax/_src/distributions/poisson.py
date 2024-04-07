from __future__ import annotations

from typing import Any, Self

import jax
from jax.scipy import special as jss
from tjax import JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.conjugate_prior import HasConjugatePrior
from ..interfaces.samplable import Samplable
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter, integral_ring, positive_support
from ..parametrization import SimpleDistribution
from .gamma import GammaNP


@dataclass
class PoissonNP(Samplable,
                NaturalParametrization['PoissonEP', JaxRealArray],
                SimpleDistribution):
    """The natural parametrization of the Poisson distribution.

    Args:
        log_mean: log(E(x)).
    """
    log_mean: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.log_mean.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=integral_ring)

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.get_namespace()
        return xp.exp(self.log_mean)

    @override
    def to_exp(self) -> PoissonEP:
        xp = self.get_namespace()
        return PoissonEP(xp.exp(self.log_mean))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return -jss.gammaln(x + 1)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> PoissonEP:
        return PoissonEP(x)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)


@dataclass
class PoissonEP(HasConjugatePrior,
                Samplable,
                ExpectationParametrization[PoissonNP],
                SimpleDistribution):
    """The expectation parametrization of the Poisson distribution.

    Args:
        mean: E(x).
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=integral_ring)

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[PoissonNP]:
        return PoissonNP

    @override
    def to_nat(self) -> PoissonNP:
        xp = self.get_namespace()
        return PoissonNP(xp.log(self.mean))

    # The expected_carrier_measure is -exp(-mean) * sum over k from zero to infinity of
    #   lambda ** k * log(k!) / k! = lambda ** k * log Gamma(k+1) / Gamma(k+1)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        return jax.random.poisson(key, self.mean, shape)

    @override
    def conjugate_prior_distribution(self, n: JaxRealArray) -> GammaNP:
        return GammaNP(-n, n * self.mean)

    @classmethod
    @override
    def from_conjugate_prior_distribution(cls, cp: NaturalParametrization[Any, Any]
                                          ) -> tuple[Self, JaxRealArray]:
        assert isinstance(cp, GammaNP)
        n = -cp.negative_rate
        mean = cp.shape_minus_one / n
        return (cls(mean), n)

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        return self.mean
