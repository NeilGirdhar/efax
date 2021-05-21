from __future__ import annotations

from math import comb, sqrt
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from tjax import Array, Shape
from tjax.dataclasses import field

__all__ = ['Support', 'ScalarSupport', 'VectorSupport', 'SymmetricMatrixSupport',
           'SquareMatrixSupport']


class Support:
    def axes(self) -> int:
        raise NotImplementedError

    def num_elements(self, dimensions: int) -> int:
        raise NotImplementedError

    def shape(self, dimensions: int) -> Shape:
        raise NotImplementedError

    def flattened(self, x: Array) -> Array:
        raise NotImplementedError

    def unflattened(self, x: Array, dimensions: int) -> Array:
        raise NotImplementedError


class ScalarSupport(Support):
    def axes(self) -> int:
        return 0

    def num_elements(self, dimensions: int) -> int:
        return 1

    def shape(self, dimensions: int) -> Shape:
        return ()

    def flattened(self, x: Array) -> Array:
        return jnp.reshape(x, (*x.shape, 1))

    def unflattened(self, x: Array, dimensions: int) -> Array:
        assert x.shape[-1] == 1
        return jnp.reshape(x, x.shape[:-1])


class VectorSupport(Support):
    def axes(self) -> int:
        return 1

    def num_elements(self, dimensions: int) -> int:
        return dimensions

    def shape(self, dimensions: int) -> Shape:
        return (dimensions,)

    def flattened(self, x: Array) -> Array:
        return x

    def unflattened(self, x: Array, dimensions: int) -> Array:
        assert x.shape[-1] == dimensions
        return x


class SymmetricMatrixSupport(Support):
    def __init__(self, *, hermitian: bool = False):
        self.hermitian = hermitian

    def axes(self) -> int:
        return 2

    def num_elements(self, dimensions: int) -> int:
        return comb(dimensions + 1, 2)

    def shape(self, dimensions: int) -> Shape:
        return (dimensions, dimensions)

    def flattened(self, x: Array) -> Array:
        dimensions = x.shape[-1]
        assert x.shape[-2] == dimensions
        index = (..., *jnp.triu_indices(dimensions))
        return x[index]

    def unflattened(self, x: Array, dimensions: int) -> Array:
        k = x.shape[-1]
        sqrt_discriminant = sqrt(1 + 8 * k)
        i_sqrt_discriminant = int(sqrt_discriminant)
        if i_sqrt_discriminant != sqrt_discriminant:
            raise ValueError(f"{k} {sqrt_discriminant}")
        if i_sqrt_discriminant % 2 != 1:
            raise ValueError
        dimensions = (i_sqrt_discriminant - 1) // 2
        index = (..., *jnp.triu_indices(dimensions))
        empty = jnp.empty(x.shape[:-1] + (dimensions, dimensions))
        lower_diagonal = empty.at[index].set(x).T
        if self.hermitian:
            lower_diagonal = jnp.conjugate(lower_diagonal)
        return lower_diagonal.at[index].set(x)


class SquareMatrixSupport(Support):
    def axes(self) -> int:
        return 2

    def num_elements(self, dimensions: int) -> int:
        return dimensions ** 2

    def shape(self, dimensions: int) -> Shape:
        return (dimensions, dimensions)

    def flattened(self, x: Array) -> Array:
        return jnp.reshape(x, (*x.shape[:-2], -1))

    def unflattened(self, x: Array, dimensions: int) -> Array:
        return jnp.reshape(x, x.shape[:-1] + self.shape(dimensions))


def distribution_parameter(support: Support, fixed: bool = False, static: bool = False) -> Any:
    if static and not fixed:
        raise ValueError
    return field(static=static, metadata={'support': support, 'fixed': fixed})


if TYPE_CHECKING:
    pass
