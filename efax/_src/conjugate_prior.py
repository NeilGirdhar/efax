from __future__ import annotations

from typing import Any, Generic

from tjax import ComplexArray, RealArray

from .expectation_parametrization import NP, ExpectationParametrization
from .multidimensional import Multidimensional
from .natural_parametrization import NaturalParametrization

__all__ = ['HasConjugatePrior', 'HasGeneralizedConjugatePrior']


class HasConjugatePrior(ExpectationParametrization[NP], Generic[NP]):
    # Abstract methods -----------------------------------------------------------------------------
    def conjugate_prior_distribution(self, n: RealArray) -> NaturalParametrization[Any, Any]:
        """The conjugate prior distribution.

        Args:
            n: The nonnegative number of pseudo-observations.  Must have shape == self.shape.
        """
        raise NotImplementedError

    def conjugate_prior_observation(self) -> ComplexArray:
        """An observation of the conjugate prior corresponding to this distribution."""
        raise NotImplementedError


class HasGeneralizedConjugatePrior(HasConjugatePrior[NP], Multidimensional, Generic[NP]):
    # Abstract methods -----------------------------------------------------------------------------
    def generalized_conjugate_prior_distribution(self, n: RealArray
                                                 ) -> NaturalParametrization[Any, Any]:
        """A generalization of the conjugate prior distribution.

        Args:
            n: The nonnegative number of pseudo-observations.  Must have
                shape == (*self.shape, self.dimensions()).
        """
        raise NotImplementedError
