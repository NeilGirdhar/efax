from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.nn import softplus
from jax.random import KeyArray
from jax.scipy import special as jss
from tjax import JaxRealArray, Shape, inverse_softplus
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.samplable import Samplable
from ..mixins.exp_to_nat import ExpToNat
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter

__all__ = ['GammaNP', 'GammaEP']


@dataclass
class GammaNP(HasEntropyNP, NaturalParametrization['GammaEP', JaxRealArray], Samplable):
    negative_rate: JaxRealArray = distribution_parameter(ScalarSupport())
    shape_minus_one: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.negative_rate.shape

    @override
    def domain_support(self) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        shape = self.shape_minus_one + 1.0
        return jss.gammaln(shape) - shape * jnp.log(-self.negative_rate)

    @override
    def to_exp(self) -> GammaEP:
        shape = self.shape_minus_one + 1.0
        return GammaEP(-shape / self.negative_rate,
                       jss.digamma(shape) - jnp.log(-self.negative_rate))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape)

    @override
    def sufficient_statistics(self, x: JaxRealArray) -> GammaEP:
        return GammaEP(x, jnp.log(x))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        return -jax.random.gamma(key, self.shape_minus_one + 1.0, shape) / self.negative_rate


@dataclass
class GammaEP(HasEntropyEP[GammaNP], ExpToNat[GammaNP, JaxRealArray]):
    mean: JaxRealArray = distribution_parameter(ScalarSupport())
    mean_log: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    def domain_support(self) -> ScalarSupport:
        return ScalarSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[GammaNP]:
        return GammaNP

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.shape)

    @override
    def search_to_natural(self, search_parameters: JaxRealArray) -> GammaNP:
        shape = softplus(search_parameters)
        rate = shape / self.mean
        return GammaNP(-rate, shape - 1.0)

    @override
    def search_gradient(self, search_parameters: JaxRealArray) -> JaxRealArray:
        shape = softplus(search_parameters)
        log_mean_minus_mean_log = jnp.log(self.mean) - self.mean_log
        return log_mean_minus_mean_log - jnp.log(shape) + jss.digamma(shape)
        # gradient is 1.0 / shape - jss.polygamma(1, shape)
        # where polygamma(1) is trigamma

    @override
    def initial_search_parameters(self) -> JaxRealArray:
        log_mean_minus_mean_log = jnp.log(self.mean) - self.mean_log
        initial_shape = (
            (3.0 - log_mean_minus_mean_log
             + jnp.sqrt((log_mean_minus_mean_log - 3.0) ** 2 + 24.0 * log_mean_minus_mean_log))
            / (12.0 * log_mean_minus_mean_log))
        return inverse_softplus(initial_shape)
