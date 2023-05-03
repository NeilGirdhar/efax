from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ..parametrization import Parametrization

__all__ = ['Multidimensional']


class Multidimensional(Parametrization[Any]):
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError
