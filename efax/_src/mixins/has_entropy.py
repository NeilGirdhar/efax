from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, final, override

from tjax import JaxAbstractClass, JaxRealArray, abstract_jit, jit, stop_gradient
from typing_extensions import TypeVar

from efax._src.expectation_parametrization import ExpectationParametrization
from efax._src.natural_parametrization import NaturalParametrization
from efax._src.parametrization import Distribution
from efax._src.tools import parameter_dot_product

NP = TypeVar("NP", bound=NaturalParametrization, default=Any)


class HasEntropy(Distribution):
    """A Distribution whose Shannon entropy can be computed."""

    def entropy(self) -> JaxRealArray:
        """Return the Shannon entropy of the distribution."""
        raise NotImplementedError


class HasEntropyEP(ExpectationParametrization[NP], HasEntropy, JaxAbstractClass, Generic[NP]):
    """An ExpectationParametrization with analytically tractable entropy.

    Provides entropy and cross-entropy via the inner-product form of the exponential family,
    requiring only the expected carrier measure as an additional implementation detail.
    """

    @abstract_jit
    @abstractmethod
    def expected_carrier_measure(self) -> JaxRealArray:
        """Return the expected carrier measure E[h(x)] under this distribution.

        This is the term missing from the natural-parameter inner product ⟨η, E[T(x)]⟩
        when computing entropy or cross-entropy.  Often zero (e.g. for Normal, Poisson).
        """
        raise NotImplementedError

    @jit
    @final
    def cross_entropy(self, q: NP) -> JaxRealArray:
        """Return the cross-entropy H(self, q) = -E_self[log q(x)].

        Args:
            q: The natural parameters of the predictive distribution.
        """
        return (
            -parameter_dot_product(q, self) + q.log_normalizer() - self.expected_carrier_measure()
        )

    @jit
    @final
    @override
    def entropy(self) -> JaxRealArray:
        """Return the Shannon entropy H(self) = -E_self[log self(x)].

        Computed as self.cross_entropy(self.to_nat()).  Can be slow for distributions
        that require numerical inversion to obtain natural parameters.
        """
        return self.cross_entropy(stop_gradient(self.to_nat()))


EP = TypeVar("EP", bound=HasEntropyEP, default=Any)


class HasEntropyNP(NaturalParametrization[EP], HasEntropy, Generic[EP]):
    """A NaturalParametrization with analytically tractable entropy.

    Delegates to the corresponding expectation parametrization's cross_entropy.
    """

    @jit
    @final
    @override
    def entropy(self) -> JaxRealArray:
        """Return the Shannon entropy H(self) = -E_self[log self(x)]."""
        return self.to_exp().cross_entropy(self)
