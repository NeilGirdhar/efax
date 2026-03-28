from __future__ import annotations

from abc import abstractmethod

from efax._src.parametrization import SimpleDistribution


class Multidimensional(SimpleDistribution):
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError
