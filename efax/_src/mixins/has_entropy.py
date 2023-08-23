from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar, final

from tjax import JaxAbstractClass, JaxRealArray, abstract_jit, jit

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import NaturalParametrization
from ..tools import parameters_dot_product

__all__ = ['HasEntropyEP', 'HasEntropyNP']


NP = TypeVar('NP', bound=NaturalParametrization[Any, Any])
class HasEntropyEP(ExpectationParametrization[NP],
                   JaxAbstractClass,
                   Generic[NP]):
    @abstract_jit
    @abstractmethod
    def expected_carrier_measure(self) -> JaxRealArray:
        """The expected carrier measure of the distribution.

        This is the missing term from the inner product between the observed distribution and the
        predicted distribution.  Often, it is just jnp.zeros(self.shape).
        """
        raise NotImplementedError

    @jit
    @final
    def cross_entropy(self, q: NP) -> JaxRealArray:
        """The cross entropy.

        Args:
            q: The natural parameters of the prediction.
        """
        return (-parameters_dot_product(q, self)
                + q.log_normalizer()
                - self.expected_carrier_measure())

    @jit
    @final
    def entropy(self) -> JaxRealArray:
        """The Shannon entropy.

        This can be quite slow since it depends on a conversion to natural parameters.
        """
        return self.cross_entropy(self.to_nat())


EP = TypeVar('EP', bound=HasEntropyEP[Any])
class HasEntropyNP(NaturalParametrization[EP, Any],
                   Generic[EP]):
    @jit
    @final
    def entropy(self) -> JaxRealArray:
        """The Shannon entropy."""
        return self.to_exp().cross_entropy(self)
