from __future__ import annotations

from abc import abstractmethod

from tjax import JaxArray, KeyArray, Shape

from efax._src.parametrization import SimpleDistribution


class Samplable(SimpleDistribution):
    @abstractmethod
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxArray:
        raise NotImplementedError
