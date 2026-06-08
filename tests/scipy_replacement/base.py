from __future__ import annotations

from typing import Protocol, runtime_checkable

from numpy.random import Generator
from tjax import NumpyComplexArray, NumpyIntegralArray, NumpyRealArray, Shape


class ScipyDistribution:
    def pdf(self, x: NumpyRealArray) -> NumpyRealArray:
        raise NotImplementedError

    def sample(self, shape: Shape = (), *, rng: Generator | None = None) -> NumpyRealArray:
        raise NotImplementedError

    def entropy(self) -> NumpyRealArray:
        raise NotImplementedError


class ScipyComplexDistribution:
    def pdf(self, x: NumpyComplexArray) -> NumpyRealArray:
        raise NotImplementedError

    def sample(self, shape: Shape = (), *, rng: Generator | None = None) -> NumpyComplexArray:
        raise NotImplementedError

    def entropy(self) -> NumpyRealArray:
        raise NotImplementedError


class ScipyDiscreteDistribution:
    def pmf(self, x: NumpyIntegralArray) -> NumpyRealArray:
        raise NotImplementedError

    def sample(self, shape: Shape = (), *, rng: Generator | None = None) -> NumpyIntegralArray:
        raise NotImplementedError

    def entropy(self) -> NumpyRealArray:
        raise NotImplementedError


@runtime_checkable
class OldStyleScipyDistribution(Protocol):
    def pdf(self, x: NumpyComplexArray) -> NumpyRealArray: ...

    def rvs(self, size: Shape = (), *, random_state: Generator | None = None) -> NumpyRealArray: ...

    def entropy(self) -> NumpyRealArray: ...


@runtime_checkable
class OldStyleScipyDiscreteDistribution(Protocol):
    def pmf(self, x: NumpyIntegralArray) -> NumpyRealArray: ...

    def rvs(
        self, size: Shape = (), *, random_state: Generator | None = None
    ) -> NumpyIntegralArray: ...

    def entropy(self) -> NumpyRealArray: ...
