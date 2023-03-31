from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from tjax import JaxComplexArray, JaxRealArray, Shape
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import EP, NaturalParametrization

__all__: list[str] = []


TEP = TypeVar('TEP', bound=ExpectationParametrization[Any])
NP = TypeVar('NP', bound=NaturalParametrization[Any, Any])
Domain = TypeVar('Domain', bound=JaxComplexArray)


class TransformedNaturalParametrization(NaturalParametrization[TEP, Domain],
                                        Generic[NP, EP, TEP, Domain]):
    """Produce a NaturalParametrization by relating it to some base distrubtion NP."""
    @abstractmethod
    def base_distribution(self) -> NP:
        raise NotImplementedError

    @abstractmethod
    def create_expectation(self, expectation_parametrization: EP) -> TEP:
        raise NotImplementedError

    @abstractmethod
    def sample_to_base_sample(self, x: Domain) -> Domain:
        raise NotImplementedError

    @property
    @override
    def shape(self) -> Shape:
        return self.base_distribution().shape

    @override
    def log_normalizer(self) -> JaxRealArray:
        """The log-normalizer."""
        return self.base_distribution().log_normalizer()

    @override
    def to_exp(self) -> TEP:
        """The corresponding expectation parameters."""
        return self.create_expectation(self.base_distribution().to_exp())

    @override
    def sufficient_statistics(self, x: Domain) -> TEP:
        y = self.sample_to_base_sample(x)
        return self.create_expectation(self.base_distribution().sufficient_statistics(y))


TNP = TypeVar('TNP', bound=TransformedNaturalParametrization[Any, Any, Any, Any])


class TransformedExpectationParametrization(ExpectationParametrization[TNP], Generic[EP, NP, TNP]):
    """Produce an ExpectationParametrization by relating it to some base distrubtion EP."""
    @abstractmethod
    def base_distribution(self) -> EP:
        raise NotImplementedError

    @abstractmethod
    def create_natural(self, natural_parametrization: NP) -> TNP:
        raise NotImplementedError

    @property
    @override
    def shape(self) -> Shape:
        return self.base_distribution().shape

    @override
    def to_nat(self) -> TNP:
        """The corresponding natural parameters."""
        return self.create_natural(self.base_distribution().to_nat())
