from __future__ import annotations

from abc import abstractmethod
from typing import Any

from tjax import JaxComplexArray, JaxRealArray

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.multidimensional import Multidimensional
from ..natural_parametrization import NaturalParametrization

__all__ = ['HasConjugatePrior', 'HasGeneralizedConjugatePrior']


class HasConjugatePrior(ExpectationParametrization[Any]):
    @abstractmethod
    def conjugate_prior_distribution(self, n: JaxRealArray
                                     ) -> NaturalParametrization[Any, Any]:
        """The conjugate prior distribution.

        Args:
            n: The nonnegative number of pseudo-observations.  Must have shape == self.shape.
        """
        raise NotImplementedError

    @abstractmethod
    def conjugate_prior_observation(self) -> JaxComplexArray:
        """An observation of the conjugate prior corresponding to this distribution."""
        raise NotImplementedError


class HasGeneralizedConjugatePrior(HasConjugatePrior, Multidimensional):
    @abstractmethod
    def generalized_conjugate_prior_distribution(self, n: JaxRealArray
                                                 ) -> NaturalParametrization[Any, Any]:
        """A generalization of the conjugate prior distribution.

        Args:
            n: The nonnegative number of pseudo-observations.  Must have
                shape == (*self.shape, self.dimensions()).
        """
        raise NotImplementedError
