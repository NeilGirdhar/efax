from __future__ import annotations

import math
from typing import Optional, Type

import jax
import jax.numpy as jnp
from tjax import Generator, RealArray, Shape
from tjax.dataclasses import dataclass

from ...conjugate_prior import HasConjugatePrior
from ...natural_parametrization import NaturalParametrization
from ...parameter import VectorSupport, distribution_parameter
from ...samplable import Samplable
from .isotropic import IsotropicNormalNP

__all__ = ['MultivariateUnitNormalNP', 'MultivariateUnitNormalEP']


@dataclass
class MultivariateUnitNormalNP(NaturalParametrization['MultivariateUnitNormalEP', RealArray],
                               Samplable):
    """
    The multivariate normal distribution with unit variance.  This is a curved exponential family.
    """
    mean: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    def log_normalizer(self) -> RealArray:
        num_parameters = self.mean.shape[-1]
        return 0.5 * (jnp.sum(jnp.square(self.mean), axis=-1)
                      + num_parameters * math.log(math.pi * 2.0))

    def to_exp(self) -> MultivariateUnitNormalEP:
        return MultivariateUnitNormalEP(self.mean)

    def carrier_measure(self, x: RealArray) -> RealArray:
        # The second moment of a delta distribution at x.
        return -0.5 * jnp.sum(jnp.square(x), axis=-1)

    def sufficient_statistics(self, x: RealArray) -> MultivariateUnitNormalEP:
        return MultivariateUnitNormalEP(x)

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> RealArray:
        if shape is not None:
            shape += self.mean.shape
        else:
            shape = self.mean.shape
        return jax.random.normal(rng.key, shape) + self.mean

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.mean.shape[-1]


@dataclass
class MultivariateUnitNormalEP(HasConjugatePrior[MultivariateUnitNormalNP], Samplable):
    mean: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @classmethod
    def natural_parametrization_cls(cls) -> Type[MultivariateUnitNormalNP]:
        return MultivariateUnitNormalNP

    def to_nat(self) -> MultivariateUnitNormalNP:
        return MultivariateUnitNormalNP(self.mean)

    def expected_carrier_measure(self) -> RealArray:
        num_parameters = self.mean.shape[-1]
        # The second moment of a normal distribution with the given mean.
        return -0.5 * (jnp.sum(jnp.square(self.mean), axis=-1) + num_parameters)

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> RealArray:
        return self.to_nat().sample(rng, shape)

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    # Overridden methods ---------------------------------------------------------------------------
    def conjugate_prior_distribution(self, n: RealArray) -> IsotropicNormalNP:
        negative_half_precision = -0.5 * n * jnp.ones(self.shape)
        return IsotropicNormalNP(n[..., jnp.newaxis] * self.mean, negative_half_precision)

    def conjugate_prior_observation(self) -> RealArray:
        return self.mean
