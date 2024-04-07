from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar, final, override

from jax.lax import stop_gradient
from tjax import JaxAbstractClass, JaxRealArray, abstract_jit, jit

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import NaturalParametrization
from ..parametrization import Distribution
from ..tools import parameter_dot_product

NP = TypeVar('NP', bound=NaturalParametrization[Any, Any])


class HasEntropy(Distribution):
    def entropy(self) -> JaxRealArray:
        raise NotImplementedError


class HasEntropyEP(ExpectationParametrization[NP],
                   HasEntropy,
                   JaxAbstractClass,
                   Generic[NP]):
    @abstract_jit
    @abstractmethod
    def expected_carrier_measure(self) -> JaxRealArray:
        """The expected carrier measure of the distribution.

        This is the missing term from the inner product between the observed distribution and the
        predicted distribution.  Often, it is just xp.zeros(self.shape).
        """
        raise NotImplementedError

    @jit
    @final
    def cross_entropy(self, q: NP) -> JaxRealArray:
        """The cross entropy.

        Args:
            q: The natural parameters of the prediction.
        """
        return (-parameter_dot_product(q, self)
                + q.log_normalizer()
                - self.expected_carrier_measure())

    @jit
    @final
    @override
    def entropy(self) -> JaxRealArray:
        """The Shannon entropy.

        This can be quite slow since it depends on a conversion to natural parameters.
        """
        return self.cross_entropy(stop_gradient(self.to_nat()))


EP = TypeVar('EP', bound=HasEntropyEP[Any])


class HasEntropyNP(NaturalParametrization[EP, Any],
                   HasEntropy,
                   Generic[EP]):
    @jit
    @final
    @override
    def entropy(self) -> JaxRealArray:
        """The Shannon entropy."""
        return self.to_exp().cross_entropy(self)
