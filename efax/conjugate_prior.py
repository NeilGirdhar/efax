from typing import Any, Generic

from tjax import Array, RealArray

from .exponential_family import NP, ExpectationParametrization, NaturalParametrization

__all__ = ['HasConjugatePrior']


class HasConjugatePrior(ExpectationParametrization[NP], Generic[NP]):

    # Abstract methods -----------------------------------------------------------------------------
    def conjugate_prior_distribution(self, n: RealArray) -> NaturalParametrization[Any]:
        raise NotImplementedError

    def conjugate_prior_observation(self) -> Array:
        raise NotImplementedError
