from __future__ import annotations

from jax.random import KeyArray
from tjax import Array, Shape

from .parametrization import Parametrization

__all__ = ['Samplable']


class Samplable(Parametrization):
    def sample(self, key: KeyArray, shape: Shape | None = None) -> Array:
        raise NotImplementedError
