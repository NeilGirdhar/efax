from __future__ import annotations

from abc import abstractmethod
from typing import Self

from tjax import JaxComplexArray, JaxRealArray

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.multidimensional import Multidimensional
from ..natural_parametrization import NaturalParametrization


class HasConjugatePrior(ExpectationParametrization):
    @abstractmethod
    def conjugate_prior_distribution(self, n: JaxRealArray
                                     ) -> NaturalParametrization:
        """The conjugate prior distribution.

        Args:
            n: The nonnegative number of pseudo-observations.  Must have shape == self.shape.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_conjugate_prior_distribution(cls, cp: NaturalParametrization
                                          ) -> tuple[Self, JaxRealArray]:
        """Given a conjugate prior distribution, find the distribution and observation count.

        Args:
            cp: The conjugate prior distribution.

        Returns:
            The distribution that gave rise to the conjguate prior, and the observation count.
        """
        raise NotImplementedError

    @abstractmethod
    def conjugate_prior_observation(self) -> JaxComplexArray:
        """An observation of the conjugate prior corresponding to this distribution."""
        raise NotImplementedError


class HasGeneralizedConjugatePrior(HasConjugatePrior, Multidimensional):
    @abstractmethod
    def generalized_conjugate_prior_distribution(self, n: JaxRealArray
                                                 ) -> NaturalParametrization:
        """A generalization of the conjugate prior distribution.

        Args:
            n: The nonnegative number of pseudo-observations.  Must have
                shape == (*self.shape, self.dimensions()).
        """
        raise NotImplementedError
