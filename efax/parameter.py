from __future__ import annotations

from math import comb, sqrt
from typing import TYPE_CHECKING, Any, Iterable, Tuple, Type, Union

import jax.numpy as jnp
from jax.ops import index_update
from tjax import Array, Shape, field, field_names_values_metadata, fields

__all__ = ['parameter_names_values_support', 'parameter_names_support', 'Support', 'ScalarSupport',
           'VectorSupport', 'SymmetricMatrixSupport', 'SquareMatrixSupport']


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
        lower_diagonal = index_update(empty, index, x).T
        if self.hermitian:
            lower_diagonal = jnp.conjugate(lower_diagonal)
        return index_update(lower_diagonal, index, x)


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


def distribution_parameter(support: Support) -> Any:
    return field(metadata={'support': support})


def parameter_names_values_support(x: Parametrization) -> Iterable[Tuple[str, Array, Support]]:
    for name, value, metadata in field_names_values_metadata(x, static=False):
        support = metadata['support']
        if not isinstance(support, Support):
            raise TypeError
        yield name, value, support


def parameter_names_support(x: Union[Type[Parametrization], Parametrization]) -> (
        Iterable[Tuple[str, Support]]):
    for this_field in fields(x, static=False):
        support = this_field.metadata['support']
        if not isinstance(support, Support):
            raise TypeError
        yield this_field.name, support


if TYPE_CHECKING:
    from .expectation_parametrization import Parametrization
