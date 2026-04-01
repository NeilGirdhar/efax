from __future__ import annotations

from abc import abstractmethod
from typing import Self

from tjax import JaxComplexArray, JaxRealArray

from efax._src.expectation_parametrization import ExpectationParametrization
from efax._src.interfaces.multidimensional import Multidimensional
from efax._src.natural_parametrization import NaturalParametrization


class HasConjugatePrior(ExpectationParametrization):
    """An ExpectationParametrization whose natural conjugate prior is known analytically.

    The conjugate prior of a distribution in an exponential family is itself a distribution
    whose sufficient statistics are the natural parameters and log-normalizer of the likelihood.
    Implementing this interface enables Bayesian updates in closed form.
    """

    @abstractmethod
    def conjugate_prior_distribution(self, n: JaxRealArray) -> NaturalParametrization:
        """Return the conjugate prior distribution centred on this distribution.

        Args:
            n: The nonnegative pseudo-observation count.  Must have shape == self.shape.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_conjugate_prior_distribution(
        cls, cp: NaturalParametrization
    ) -> tuple[Self, JaxRealArray]:
        """Recover the distribution and observation count encoded in a conjugate prior.

        Args:
            cp: The conjugate prior distribution.

        Returns:
            The distribution that gave rise to the conjugate prior, and the observation count.
        """
        raise NotImplementedError

    @abstractmethod
    def as_conjugate_prior_observation(self) -> JaxComplexArray:
        """Return this distribution expressed as an observation of its own conjugate prior.

        This is the sufficient statistic of the conjugate prior that corresponds to the
        current distribution's parameters — i.e. the value cp_x such that updating a
        conjugate prior with cp_x moves it towards self.
        """
        raise NotImplementedError


class HasGeneralizedConjugatePrior(HasConjugatePrior, Multidimensional):
    """A HasConjugatePrior for multidimensional distributions.

    Extends HasConjugatePrior with a generalized conjugate prior whose pseudo-observation
    count has shape (*self.shape, self.dimensions()) rather than self.shape, allowing
    independent pseudo-observation counts along each dimension.
    """

    @abstractmethod
    def generalized_conjugate_prior_distribution(self, n: JaxRealArray) -> NaturalParametrization:
        """Return the generalized conjugate prior distribution centred on this distribution.

        Args:
            n: The nonnegative pseudo-observation counts.
                Must have shape == (*self.shape, self.dimensions()).
        """
        raise NotImplementedError
