from __future__ import annotations

from abc import abstractmethod

from tjax import JaxArray, KeyArray, Shape

from ..parametrization import Parametrization

__all__ = ['Samplable']


class Samplable(Parametrization):
    @abstractmethod
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxArray:
        raise NotImplementedError
