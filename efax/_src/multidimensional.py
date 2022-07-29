from __future__ import annotations

from .parametrization import Parametrization

__all__ = ['Multidimensional']


class Multidimensional(Parametrization):
    def dimensions(self) -> int:
        raise NotImplementedError
