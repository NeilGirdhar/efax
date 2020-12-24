from __future__ import annotations

import math
from typing import Iterable

from jax import numpy as jnp
from tjax import RealArray, Shape, dataclass

from .conjugate_prior import HasConjugatePrior
from .exponential_family import NaturalParametrization
from .isotropic_normal import IsotropicNormalNP

__all__ = ['MultivariateUnitNormalNP', 'MultivariateUnitNormalEP']


@dataclass
class MultivariateUnitNormalNP(NaturalParametrization['MultivariateUnitNormalEP']):
    """
    The multivariate normal distribution with unit variance.  This is a curved exponential family.
    """
    mean: RealArray

    # Implemented methods --------------------------------------------------------------------------
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

    @classmethod
    def field_axes(cls) -> Iterable[int]:
        yield 1


@dataclass
class MultivariateUnitNormalEP(HasConjugatePrior[MultivariateUnitNormalNP]):
    mean: RealArray

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    def to_nat(self) -> MultivariateUnitNormalNP:
        return MultivariateUnitNormalNP(self.mean)

    def expected_carrier_measure(self) -> RealArray:
        num_parameters = self.mean.shape[-1]
        # The second moment of a normal distribution with the given mean.
        return -0.5 * (jnp.sum(jnp.square(self.mean), axis=-1) + num_parameters)

    # Overridden methods ---------------------------------------------------------------------------
    def conjugate_prior_distribution(self, n: RealArray) -> IsotropicNormalNP:
        negative_half_precision = -0.5 * n * jnp.ones(self.shape())
        return IsotropicNormalNP(n * self.mean, negative_half_precision)

    def conjugate_prior_observation(self) -> RealArray:
        return self.mean
