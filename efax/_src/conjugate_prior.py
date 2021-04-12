from typing import Any, Generic

from tjax import Array, RealArray

from .expectation_parametrization import NP, ExpectationParametrization
from .natural_parametrization import NaturalParametrization

__all__ = ['HasConjugatePrior']


class HasConjugatePrior(ExpectationParametrization[NP], Generic[NP]):

    # Abstract methods -----------------------------------------------------------------------------
    def conjugate_prior_distribution(self, n: RealArray) -> NaturalParametrization[Any]:
        raise NotImplementedError

    def conjugate_prior_observation(self) -> Array:
        raise NotImplementedError
