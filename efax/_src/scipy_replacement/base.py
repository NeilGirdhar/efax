from __future__ import annotations

from typing import Protocol, runtime_checkable

from numpy.random import Generator
from tjax import NumpyComplexArray, NumpyIntegralArray, NumpyRealArray, ShapeLike


@runtime_checkable
class ScipyDistribution(Protocol):
    def pdf(self, x: NumpyComplexArray) -> NumpyRealArray:
        ...

    def rvs(self, size: ShapeLike = (), random_state: Generator | None = None) -> NumpyComplexArray:
        ...

    def entropy(self) -> NumpyRealArray:
        ...


@runtime_checkable
class ScipyDiscreteDistribution(Protocol):
    def pmf(self, x: NumpyIntegralArray) -> NumpyRealArray:
        ...

    def rvs(self, size: ShapeLike = (), random_state: Generator | None = None
            ) -> NumpyIntegralArray:
        ...

    def entropy(self) -> NumpyRealArray:
        ...
