from __future__ import annotations

from math import comb, sqrt
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from tjax import Array, RealArray, Shape
from tjax.dataclasses import field

__all__ = ['Support', 'ScalarSupport', 'VectorSupport', 'SymmetricMatrixSupport',
           'SquareMatrixSupport']


class Field:
    def num_elements(self, support_num_element: int) -> int:
        raise NotImplementedError

    def flattened(self, x: Array) -> RealArray:
        raise NotImplementedError

    def unflattened(self, y: RealArray) -> Array:
        raise NotImplementedError


class RealField(Field):
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element

    def flattened(self, x: Array) -> RealArray:
        return x

    def unflattened(self, y: RealArray) -> Array:
        return y


class ComplexField(Field):
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element * 2

    def flattened(self, x: Array) -> RealArray:
        return jnp.concatenate([x.real, x.imag], axis=-1)

    def unflattened(self, y: RealArray) -> Array:
        assert y.shape[-1] % 2 == 0
        n = y.shape[-1] // 2
        return y[..., :n] + 1j * y[..., n:]


real_field = RealField()
complex_field = ComplexField()


class Support:
    def __init__(self, *, is_complex: bool = False):
        super().__init__()
        self.field = complex_field if is_complex else real_field

    def axes(self) -> int:
        raise NotImplementedError

    def shape(self, dimensions: int) -> Shape:
        raise NotImplementedError

    def num_elements(self, dimensions: int) -> int:
        raise NotImplementedError

    def flattened(self, x: Array) -> RealArray:
        raise NotImplementedError

    def unflattened(self, y: RealArray, dimensions: int) -> Array:
        raise NotImplementedError


class ScalarSupport(Support):
    def axes(self) -> int:
        return 0

    def shape(self, dimensions: int) -> Shape:
        return ()

    def num_elements(self, dimensions: int) -> int:
        return self.field.num_elements(1)

    def flattened(self, x: Array) -> RealArray:
        return self.field.flattened(jnp.reshape(x, (*x.shape, 1)))

    def unflattened(self, y: RealArray, dimensions: int) -> Array:
        x = self.field.unflattened(y)
        assert x.shape[-1] == 1
        return jnp.reshape(x, x.shape[:-1])


class VectorSupport(Support):
    def axes(self) -> int:
        return 1

    def shape(self, dimensions: int) -> Shape:
        return (dimensions,)

    def num_elements(self, dimensions: int) -> int:
        return self.field.num_elements(dimensions)

    def flattened(self, x: Array) -> RealArray:
        return self.field.flattened(x)

    def unflattened(self, y: RealArray, dimensions: int) -> Array:
        x = self.field.unflattened(y)
        assert x.shape[-1] == dimensions
        return x


class SymmetricMatrixSupport(Support):
    def __init__(self, *, hermitian: bool = False, **kwargs: Any):
        if hermitian:
            kwargs.setdefault('is_complex', True)
        super().__init__(**kwargs)
        self.hermitian = hermitian

    def axes(self) -> int:
        return 2

    def shape(self, dimensions: int) -> Shape:
        return (dimensions, dimensions)

    def num_elements(self, dimensions: int) -> int:
        return self.field.num_elements(comb(dimensions + 1, 2))

    def flattened(self, x: Array) -> RealArray:
        dimensions = x.shape[-1]
        assert x.shape[-2] == dimensions
        index = (..., *jnp.triu_indices(dimensions))
        return self.field.flattened(x[index])

    def unflattened(self, y: RealArray, dimensions: int) -> Array:
        x = self.field.unflattened(y)
        k = x.shape[-1]
        sqrt_discriminant = sqrt(1 + 8 * k)
        i_sqrt_discriminant = int(sqrt_discriminant)
        if i_sqrt_discriminant != sqrt_discriminant:
            raise ValueError(f"{k} {sqrt_discriminant}")
        if i_sqrt_discriminant % 2 != 1:
            raise ValueError
        dimensions = (i_sqrt_discriminant - 1) // 2
        index = (..., *jnp.triu_indices(dimensions))
        empty = jnp.empty(x.shape[:-1] + (dimensions, dimensions), dtype=x.dtype)
        lower_diagonal = jnp.swapaxes(empty.at[index].set(x), -2, -1)
        if self.hermitian:
            lower_diagonal = lower_diagonal.conjugate()
        return lower_diagonal.at[index].set(x)


class SquareMatrixSupport(Support):
    def axes(self) -> int:
        return 2

    def shape(self, dimensions: int) -> Shape:
        return (dimensions, dimensions)

    def num_elements(self, dimensions: int) -> int:
        return self.field.num_elements(dimensions ** 2)

    def flattened(self, x: Array) -> Array:
        y = jnp.reshape(x, (*x.shape[:-2], -1))
        return self.field.flattened(y)

    def unflattened(self, y: Array, dimensions: int) -> Array:
        x = self.field.unflattened(y)
        return jnp.reshape(x, x.shape[:-1] + self.shape(dimensions))


def distribution_parameter(support: Support, fixed: bool = False, static: bool = False) -> Any:
    if static and not fixed:
        raise ValueError
    return field(static=static, metadata={'support': support, 'fixed': fixed})


if TYPE_CHECKING:
    pass
