from __future__ import annotations

from typing import Any, Generic, List, TypeVar

from tjax import ComplexArray, RealArray, Shape

from .expectation_parametrization import ExpectationParametrization
from .natural_parametrization import EP, NaturalParametrization

__all__: List[str] = []


TEP = TypeVar('TEP', bound=ExpectationParametrization[Any])
NP = TypeVar('NP', bound=NaturalParametrization[Any, Any])
Domain = TypeVar('Domain', bound=ComplexArray)


class TransformedNaturalParametrization(NaturalParametrization[TEP, Domain],
                                        Generic[NP, EP, TEP, Domain]):
    """
    Produce a NaturalParametrization by relating it to some base distrubtion NP.
    """
    # Abstract methods -----------------------------------------------------------------------------
    def base_distribution(self) -> NP:
        raise NotImplementedError

    def create_expectation(self, expectation_parametrization: EP) -> TEP:
        raise NotImplementedError

    def sample_to_base_sample(self, x: Domain) -> Domain:
        raise NotImplementedError

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.base_distribution().shape

    def log_normalizer(self) -> RealArray:
        """
        Returns: The log-normalizer.
        """
        return self.base_distribution().log_normalizer()

    def to_exp(self) -> TEP:
        """
        Returns: The corresponding expectation parameters.
        """
        return self.create_expectation(self.base_distribution().to_exp())

    def sufficient_statistics(self, x: Domain) -> TEP:
        y = self.sample_to_base_sample(x)
        return self.create_expectation(self.base_distribution().sufficient_statistics(y))


TNP = TypeVar('TNP', bound=TransformedNaturalParametrization[Any, Any, Any, Any])


class TransformedExpectationParametrization(ExpectationParametrization[TNP], Generic[EP, NP, TNP]):
    """
    Produce an ExpectationParametrization by relating it to some base distrubtion EP.
    """
    # Abstract methods -----------------------------------------------------------------------------
    def base_distribution(self) -> EP:
        raise NotImplementedError

    def create_natural(self, natural_parametrization: NP) -> TNP:
        raise NotImplementedError

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.base_distribution().shape

    def to_nat(self) -> TNP:
        """
        Returns: The corresponding natural parameters.
        """
        return self.create_natural(self.base_distribution().to_nat())
