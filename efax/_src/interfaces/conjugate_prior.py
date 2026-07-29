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

    The ordinary conjugate prior encodes a single scalar pseudo-observation count.  The
    generalized conjugate prior (GCP) instead carries one count n per component, with shape
    (*self.shape, self.dimensions()).

    For a distribution with k expectation parameters x, the GCP is a distribution over x with
    2k natural parameters: the counts n and the scaled values diag(n) x.  The counts are the
    diagonal of the Fisher information of the expectation parameters, so a GCP is a conjugate
    prior whose precision is free per component rather than tied to a single count.

    Examples:
        - The multivariate normal distribution with isotropic variance, whose GCP is the
          multivariate normal with diagonal variance.  For (n, x), its natural parameters are
          the mean-times-precision diag(n) x and the precision diag(n).
        - The categorical distribution, whose GCP is the generalized Dirichlet distribution.
    """

    @abstractmethod
    def generalized_conjugate_prior_distribution(self, n: JaxRealArray) -> NaturalParametrization:
        """Return the generalized conjugate prior distribution centred on this distribution.

        Args:
            n: The nonnegative per-component pseudo-observation counts.
                Must have shape == (*self.shape, self.dimensions()).
        """
        raise NotImplementedError
