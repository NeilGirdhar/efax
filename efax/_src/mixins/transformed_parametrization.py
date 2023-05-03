from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from tjax import JaxComplexArray, JaxRealArray, Shape
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import EP, NaturalParametrization

__all__: list[str] = []


TEP = TypeVar('TEP', bound=ExpectationParametrization[Any, Any])
NP = TypeVar('NP', bound=NaturalParametrization[Any, Any, Any])
Domain = TypeVar('Domain', bound=JaxComplexArray)
FixedParameters = TypeVar('FixedParameters')


class TransformedNaturalParametrization(NaturalParametrization[TEP, Domain, FixedParameters],
                                        Generic[NP, EP, TEP, Domain, FixedParameters]):
    """Produce a NaturalParametrization by relating it to some base distrubtion NP."""
    @classmethod
    @abstractmethod
    def base_distribution_cls(cls) -> type[NP]:
        raise NotImplementedError

    @abstractmethod
    def base_distribution(self) -> NP:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_expectation(cls, expectation_parametrization: EP) -> TEP:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def sample_to_base_sample(cls, x: Domain, fixed_parameters: FixedParameters) -> Domain:
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
    @classmethod
    def sufficient_statistics(cls, x: Domain, fixed_parameters: FixedParameters) -> TEP:
        y = cls.sample_to_base_sample(x, fixed_parameters)
        base_cls = cls.base_distribution_cls()
        return cls.create_expectation(base_cls.sufficient_statistics(y, fixed_parameters))


TNP = TypeVar('TNP', bound=TransformedNaturalParametrization[Any, Any, Any, Any, Any])


class TransformedExpectationParametrization(ExpectationParametrization[TNP, FixedParameters],
                                            Generic[EP, NP, TNP, FixedParameters]):
    """Produce an ExpectationParametrization by relating it to some base distrubtion EP."""
    @classmethod
    @abstractmethod
    def base_distribution_cls(cls) -> type[EP]:
        raise NotImplementedError

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
