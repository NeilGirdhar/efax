from __future__ import annotations

from abc import abstractmethod
from math import comb, sqrt
from typing import Any

import jax.numpy as jnp
from tjax import JaxArray, JaxRealArray, Shape, float_dtype, int_dtype
from tjax.dataclasses import field
from typing_extensions import override

__all__ = ['Support', 'ScalarSupport', 'VectorSupport', 'SimplexSupport', 'SymmetricMatrixSupport',
           'SquareMatrixSupport', 'Field', 'RealField', 'ComplexField', 'BooleanField',
           'IntegralField']


class Field:
    @abstractmethod
    def num_elements(self, support_num_element: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def flattened(self, x: JaxArray) -> JaxRealArray:
        raise NotImplementedError

    @abstractmethod
    def unflattened(self, y: JaxRealArray) -> JaxArray:
        raise NotImplementedError


class RealField(Field):
    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        return x

    @override
    def unflattened(self, y: JaxRealArray) -> JaxArray:
        return y


class ComplexField(Field):
    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element * 2

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        return jnp.concatenate([x.real, x.imag], axis=-1)

    @override
    def unflattened(self, y: JaxRealArray) -> JaxArray:
        assert y.shape[-1] % 2 == 0
        n = y.shape[-1] // 2
        return y[..., :n] + 1j * y[..., n:]


class BooleanField(Field):
    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        return jnp.asarray(x, dtype=float_dtype)

    @override
    def unflattened(self, y: JaxRealArray) -> JaxArray:
        return jnp.asarray(y, dtype=jnp.bool_)


class IntegralField(Field):
    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        return jnp.asarray(x, dtype=float_dtype)

    @override
    def unflattened(self, y: JaxRealArray) -> JaxArray:
        return jnp.asarray(y, dtype=int_dtype)


real_field = RealField()
integral_field = IntegralField()
complex_field = ComplexField()
boolean_field = BooleanField()


class Support:
    @override
    def __init__(self, *, field: Field = real_field):
        super().__init__()
        self.field = field

    @abstractmethod
    def axes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def shape(self, dimensions: int) -> Shape:
        raise NotImplementedError

    @abstractmethod
    def num_elements(self, dimensions: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def flattened(self, x: JaxArray) -> JaxRealArray:
        raise NotImplementedError

    @abstractmethod
    def unflattened(self, y: JaxRealArray, dimensions: int) -> JaxArray:
        raise NotImplementedError


class ScalarSupport(Support):
    @override
    def axes(self) -> int:
        return 0

    @override
    def shape(self, dimensions: int) -> Shape:
        return ()

    @override
    def num_elements(self, dimensions: int) -> int:
        return self.field.num_elements(1)

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        return self.field.flattened(jnp.reshape(x, (*x.shape, 1)))

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int) -> JaxArray:
        x = self.field.unflattened(y)
        assert x.shape[-1] == 1
        return jnp.reshape(x, x.shape[:-1])


class VectorSupport(Support):
    @override
    def axes(self) -> int:
        return 1

    @override
    def shape(self, dimensions: int) -> Shape:
        return (dimensions,)

    @override
    def num_elements(self, dimensions: int) -> int:
        return self.field.num_elements(dimensions)

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        return self.field.flattened(x)

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int) -> JaxArray:
        x = self.field.unflattened(y)
        assert x.shape[-1] == dimensions
        return x


class SimplexSupport(Support):
    @override
    def axes(self) -> int:
        return 1

    @override
    def shape(self, dimensions: int) -> Shape:
        return (dimensions - 1,)

    @override
    def num_elements(self, dimensions: int) -> int:
        return self.field.num_elements(dimensions - 1)

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        return self.field.flattened(x)

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int) -> JaxArray:
        x = self.field.unflattened(y)
        assert x.shape[-1] == dimensions
        return x


class SymmetricMatrixSupport(Support):
    @override
    def __init__(self, *, hermitian: bool = False, **kwargs: Any):
        if hermitian:
            kwargs.setdefault('field', complex_field)
        super().__init__(**kwargs)
        self.hermitian = hermitian

    @override
    def axes(self) -> int:
        return 2

    @override
    def shape(self, dimensions: int) -> Shape:
        return (dimensions, dimensions)

    @override
    def num_elements(self, dimensions: int) -> int:
        return self.field.num_elements(comb(dimensions + 1, 2))

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        dimensions = x.shape[-1]
        assert x.shape[-2] == dimensions
        index = (..., *jnp.triu_indices(dimensions))
        return self.field.flattened(x[index])

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int) -> JaxArray:
        x = self.field.unflattened(y)
        k = x.shape[-1]
        sqrt_discriminant = sqrt(1 + 8 * k)
        i_sqrt_discriminant = int(sqrt_discriminant)
        if i_sqrt_discriminant != sqrt_discriminant:
            msg =f"{k} {sqrt_discriminant}"
            raise ValueError(msg)
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
    @override
    def axes(self) -> int:
        return 2

    @override
    def shape(self, dimensions: int) -> Shape:
        return (dimensions, dimensions)

    @override
    def num_elements(self, dimensions: int) -> int:
        return self.field.num_elements(dimensions ** 2)

    @override
    def flattened(self, x: JaxArray) -> JaxArray:
        y = jnp.reshape(x, (*x.shape[:-2], -1))
        return self.field.flattened(y)

    @override
    def unflattened(self, y: JaxArray, dimensions: int) -> JaxArray:
        x = self.field.unflattened(y)
        return jnp.reshape(x, x.shape[:-1] + self.shape(dimensions))


def distribution_parameter(support: Support,
                           *,
                           fixed: bool = False,
                           static: bool = False) -> Any:
    if static and not fixed:
        raise ValueError
    return field(static=static, metadata={'support': support, 'fixed': fixed})
