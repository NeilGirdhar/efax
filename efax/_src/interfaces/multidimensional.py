from __future__ import annotations

from abc import abstractmethod

from ..parametrization import SimpleDistribution


class Multidimensional(SimpleDistribution):
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError
