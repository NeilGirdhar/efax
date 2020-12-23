from __future__ import annotations

import math
from typing import Iterable

from jax import numpy as jnp
from tjax import RealArray, Shape, dataclass

from .multivariate_normal import MultivariateNormalNP
from .conjugate_prior import HasConjugatePrior
from .exponential_family import ExpectationParametrization, NaturalParametrization

__all__ = ['NormalUnitVarianceNP', 'NormalUnitVarianceEP']


@dataclass
class NormalUnitVarianceNP(NaturalParametrization['NormalUnitVarianceEP']):
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

    def to_exp(self) -> NormalUnitVarianceEP:
        return NormalUnitVarianceEP(self.mean)

    def carrier_measure(self, x: RealArray) -> RealArray:
        # The second moment of a delta distribution at x.
        return -0.5 * jnp.sum(jnp.square(x), axis=-1)

    def sufficient_statistics(self, x: RealArray) -> NormalUnitVarianceEP:
        return NormalUnitVarianceEP(x)

    @classmethod
    def field_axes(cls) -> Iterable[int]:
        yield 1


@dataclass
class NormalUnitVarianceEP(HasConjugatePrior[NormalUnitVarianceNP]):
    mean: RealArray

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    def to_nat(self) -> NormalUnitVarianceNP:
        return NormalUnitVarianceNP(self.mean)

    def expected_carrier_measure(self) -> RealArray:
        num_parameters = self.mean.shape[-1]
        # The second moment of a normal distribution with the given mean.
        return -0.5 * (jnp.sum(jnp.square(self.mean), axis=-1) + num_parameters)

    # Overridden methods ---------------------------------------------------------------------------
    def conjugate_prior_distribution(self, n: RealArray) -> NormalNP:
        num_parameters = self.mean.shape[-1]
        negative_half_precision = -0.5 * n * jnp.eye(num_parameters)
        negative_half_precision = jnp.expand_dims(negative_half_precision,
                                                  tuple(range(len(self.shape()))))
        negative_half_precision = jnp.broadcast_to(negative_half_precision,
                                                   self.shape() + (num_parameters, num_parameters))
        return MultivariateNormalNP(n * self.mean, negative_half_precision)

    def conjugate_prior_observation(self) -> Array:
        return self.mean
