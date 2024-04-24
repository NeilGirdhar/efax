from __future__ import annotations

from abc import abstractmethod

from ..parametrization import Parametrization


class Multidimensional(Parametrization):
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError
