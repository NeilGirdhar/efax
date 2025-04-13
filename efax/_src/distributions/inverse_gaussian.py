from __future__ import annotations

import jax.random as jr
from array_api_compat import array_namespace
from tjax import JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.samplable import Samplable
from ..natural_parametrization import NaturalParametrization
from ..parameter import (RealField, ScalarSupport, Support, distribution_parameter,
                         negative_support, positive_support)
from ..parametrization import SimpleDistribution


@dataclass
class InverseGaussianNP(Samplable,
                        NaturalParametrization['InverseGaussianEP', JaxRealArray],
                        SimpleDistribution):
    """The natural parametrization of the inverse Gaussian distribution.

    Args:
        negative_lambda_over_two_mu_squared: -lambda / (2 mu ** 2)
        negative_lambda_over_two: -lambda/2
    """
    negative_lambda_over_two_mu_squared: JaxRealArray = distribution_parameter(ScalarSupport(
        ring=negative_support))
    negative_lambda_over_two: JaxRealArray = distribution_parameter(ScalarSupport(
        ring=negative_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.negative_lambda_over_two_mu_squared.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.array_namespace()
        return (-0.5 * xp.log(-2.0 * self.negative_lambda_over_two)
                - 2.0 * xp.sqrt(self.negative_lambda_over_two_mu_squared
                                * self.negative_lambda_over_two))

    @override
    def to_exp(self) -> InverseGaussianEP:
        xp = self.array_namespace()
        mu = xp.sqrt(self.negative_lambda_over_two / self.negative_lambda_over_two_mu_squared)
        lambda_ = -2.0 * self.negative_lambda_over_two
        mean_reciprocal = xp.reciprocal(mu) + xp.reciprocal(lambda_)
        return InverseGaussianEP(mu, mean_reciprocal)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return -0.5 * (xp.log(2.0 * xp.pi) + 3.0 * xp.log(x))

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray
                              ) -> InverseGaussianEP:
        xp = array_namespace(x)
        return InverseGaussianEP(x, xp.reciprocal(x))

    @override
    @classmethod
    def adjust_support(cls, name: str, **kwargs: JaxArray) -> Support:
        if name != 'negative_lambda_over_two':
            return super().adjust_support(name, **kwargs)
        eta1 = kwargs['negative_lambda_over_two_mu_squared']
        xp = array_namespace(eta1)
        maximum = -xp.pow(-0.25 * eta1, 1.0 / 3.0)
        return ScalarSupport(ring=RealField(maximum=maximum))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)


@dataclass
class InverseGaussianEP(ExpectationParametrization[InverseGaussianNP],
                        Samplable,
                        SimpleDistribution):
    """The expectation parametrization of the normal distribution.

    Args:
        mean: The mean.
        mean_reciprocal: The mean of the reciprocal.
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))
    mean_reciprocal: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

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
    def natural_parametrization_cls(cls) -> type[InverseGaussianNP]:
        return InverseGaussianNP

    @override
    def to_nat(self) -> InverseGaussianNP:
        xp = self.array_namespace()
        lambda_ = xp.reciprocal(self.mean_reciprocal - xp.reciprocal(self.mean))
        eta2 = -0.5 * lambda_
        eta1 = eta2 / xp.square(self.mean)
        return InverseGaussianNP(eta1, eta2)

    @override
    @classmethod
    def adjust_support(cls, name: str, **kwargs: JaxArray) -> Support:
        if name != 'mean_reciprocal':
            return super().adjust_support(name, **kwargs)
        mean = kwargs['mean']
        xp = array_namespace(mean)
        return ScalarSupport(ring=RealField(minimum=xp.reciprocal(mean)))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        shape = self.shape if shape is None else shape + self.shape
        nu = jr.normal(key, shape)
        z = jr.uniform(key, shape)
        xp = self.array_namespace()
        y = xp.square(nu)
        mu = self.mean
        mu_squared = xp.square(mu)
        lambda_ = xp.reciprocal(self.mean_reciprocal - xp.reciprocal(self.mean))
        x = mu + 0.5 * mu_squared / lambda_ * y - (
                mu / (2.0 * lambda_)
                * xp.sqrt(4.0 * mu * lambda_ * y + mu_squared * xp.square(y)))
        return xp.where(z <= mu / (mu + x), x, mu_squared / x)
