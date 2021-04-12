from typing import Optional

from tjax import Array, Generator, Shape

from .parametrization import Parametrization

__all__ = ['Samplable']


class Samplable(Parametrization):
    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> Array:
        raise NotImplementedError
