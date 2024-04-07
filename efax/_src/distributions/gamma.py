from __future__ import annotations

from typing import Any

from array_api_compat import get_namespace
from jax.random import gamma as random_gamma
from jax.scipy import special as jss
from tjax import JaxRealArray, KeyArray, Shape, inverse_softplus, softplus
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.samplable import Samplable
from ..mixins.exp_to_nat.exp_to_nat import ExpToNat
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import (RealField, ScalarSupport, distribution_parameter, negative_support,
                         positive_support)
from ..parametrization import SimpleDistribution


@dataclass
class GammaNP(HasEntropyNP['GammaEP'],
              Samplable,
              NaturalParametrization['GammaEP', JaxRealArray],
              SimpleDistribution):
    """The natural parametrization of the Gamma distribution.

    Args:
        negative_rate: The negative rate.
        shape_minus_one: The shape minus one.
    """
    negative_rate: JaxRealArray = distribution_parameter(ScalarSupport(ring=negative_support))
    shape_minus_one: JaxRealArray = distribution_parameter(ScalarSupport(
        ring=RealField(minimum=-1.0)))

    @property
    @override
    def shape(self) -> Shape:
        return self.negative_rate.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.get_namespace()
        shape = self.shape_minus_one + 1.0
        return jss.gammaln(shape) - shape * xp.log(-self.negative_rate)

    @override
    def to_exp(self) -> GammaEP:
        xp = self.get_namespace()
        shape = self.shape_minus_one + 1.0
        return GammaEP(-shape / self.negative_rate,
                       jss.digamma(shape) - xp.log(-self.negative_rate))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.get_namespace(x)
        return xp.zeros(x.shape)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> GammaEP:
        xp = get_namespace(x)
        return GammaEP(x, xp.log(x))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        return -random_gamma(key, self.shape_minus_one + 1.0, shape) / self.negative_rate

    def to_var(self) -> GammaVP:
        rate = -self.negative_rate
        shape = self.shape_minus_one + 1.0
        mean = shape / rate
        variance = mean / rate
        return GammaVP(mean, variance)


@dataclass
class GammaEP(HasEntropyEP[GammaNP],
              Samplable,
              ExpToNat[GammaNP],
              ExpectationParametrization[GammaNP],
              SimpleDistribution):
    """The expectation parametrization of the Gamma distribution.

    Args:
        mean: The mean: E(x).
        mean_log: The mean of the log: E(log(x)).
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))
    mean_log: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[GammaNP]:
        return GammaNP

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.get_namespace()
        return xp.zeros(self.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_nat().sample(key, shape)

    @override
    def search_to_natural(self, search_parameters: JaxRealArray) -> GammaNP:
        shape = softplus(search_parameters[..., 0])
        rate = shape / self.mean
        return GammaNP(-rate, shape - 1.0)

    @override
    def search_gradient(self, search_parameters: JaxRealArray) -> JaxRealArray:
        xp = self.get_namespace()
        shape = softplus(search_parameters[..., 0])
        log_mean_minus_mean_log = xp.log(self.mean) - self.mean_log
        return (log_mean_minus_mean_log - xp.log(shape) + jss.digamma(shape))[..., xp.newaxis]
        # gradient is 1.0 / shape - jss.polygamma(1, shape)
        # where polygamma(1) is trigamma

    @override
    def initial_search_parameters(self) -> JaxRealArray:
        xp = self.get_namespace()
        log_mean_minus_mean_log = xp.log(self.mean) - self.mean_log
        initial_shape: JaxRealArray = (
            (3.0 - log_mean_minus_mean_log
             + xp.sqrt((log_mean_minus_mean_log - 3.0) ** 2 + 24.0 * log_mean_minus_mean_log))
            / (12.0 * log_mean_minus_mean_log))
        return inverse_softplus(initial_shape)[..., xp.newaxis]


@dataclass
class GammaVP(Samplable, SimpleDistribution):
    """The variance parametrization of the Gamma distribution.

    Args:
        mean: The mean.
        variance: The variance.
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))
    variance: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    def to_nat(self) -> GammaNP:
        # mean = s/r
        # var = s/r^2
        rate = self.mean / self.variance
        shape = self.mean * rate
        return GammaNP(-rate, shape - 1.0)

    def to_exp(self) -> GammaEP:
        return self.to_nat().to_exp()

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_nat().sample(key, shape)
