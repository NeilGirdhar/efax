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

    The ordinary conjugate prior encodes evidence with a single scalar pseudo-observation count n
    shared by every component of the pseudo-sufficient statistics n x.  This is appropriate for a
    point observation, but not for a distribution: a distribution can be sharp in one
    sufficient-statistic dimension and diffuse in another, so its evidence strength should vary
    across components.

    The generalized conjugate prior (GCP) captures this by assigning one count n per component,
    with shape (*self.shape, self.dimensions()).  For a distribution with k expectation
    parameters x, the GCP is a distribution over x with 2k natural parameters: the per-component
    counts n and the scaled values diag(n) x, satisfying diag(n) == Fisher(x).  The count of each
    component equals its Fisher information at that component's value, so a GCP is a conjugate
    prior whose precision is free per component rather than tied to a single scalar count. Adding
    two GCPs adds their natural parameters, implementing the standard conjugate update.

    For k = 1, the GCP reduces to the ordinary conjugate prior: a single count and a single
    scaled value.

    Examples:
        - The multivariate normal distribution with isotropic variance, whose GCP is the
          multivariate normal with diagonal variance.  For (n, x), its natural parameters are
          the mean-times-precision diag(n) x and the precision diag(n).
        - The categorical distribution, whose GCP is the generalized Dirichlet distribution
          (T.-T. Wong 1998. Generalized Dirichlet distribution in Bayesian analysis. Applied
          Mathematics and Computation, volume 97, pp165-181).
    """

    @abstractmethod
    def generalized_conjugate_prior_distribution(self, n: JaxRealArray) -> NaturalParametrization:
        """Return the generalized conjugate prior distribution centred on this distribution.

        Args:
            n: The nonnegative per-component pseudo-observation counts.
                Must have shape == (*self.shape, self.dimensions()).
        """
        raise NotImplementedError
