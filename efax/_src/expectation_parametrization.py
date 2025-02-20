from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar, final

from tjax import JaxRealArray, jit

from .natural_parametrization import NaturalParametrization
from .parametrization import Distribution
from .tools import parameter_dot_product, parameter_map

NP = TypeVar('NP', bound=NaturalParametrization[Any, Any])


class ExpectationParametrization(Distribution, Generic[NP]):
    """The expectation parametrization of an exponential family distribution.

    This class also doubles as the sufficient statistics of an exponential family distribution.

    The motivation for the expectation parametrization is combining independent and identically
    distributed observations into the maximum likelihood distribution.  In the expectation
    parametrization, this is an expected value.
    """
    @classmethod
    @abstractmethod
    def natural_parametrization_cls(cls) -> type[NP]:
        raise NotImplementedError

    @abstractmethod
    def to_nat(self) -> NP:
        """The corresponding natural parameters."""
        raise NotImplementedError

    @jit
    @final
    def kl_divergence(self, q: NP, *, self_nat: NP | None = None) -> JaxRealArray:
        """The Kullback-Leibler divergence.

        This can be quite slow since it depends on a conversion to natural parameters.

        Args:
            self: The expectation parameters of the first distribution.
            q: The natural parameters of second destribution.
            self_nat: The natural parameters of the first distribution, if available.
        """
        if self_nat is None:
            self_nat = self.to_nat()
        xp = self.array_namespace()
        difference = parameter_map(xp.subtract, self_nat, q)
        return (parameter_dot_product(difference, self)
                + q.log_normalizer()
                - self_nat.log_normalizer())
