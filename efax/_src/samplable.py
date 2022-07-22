from __future__ import annotations

from tjax import Array, Generator, Shape

from .parametrization import Parametrization

__all__ = ['Samplable']


class Samplable(Parametrization):
    def sample(self, rng: Generator, shape: Shape | None = None) -> Array:
        raise NotImplementedError
