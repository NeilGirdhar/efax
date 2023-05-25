from __future__ import annotations

from abc import abstractmethod
from math import comb, sqrt
from typing import Any

import jax.numpy as jnp
from jax.dtypes import canonicalize_dtype
from tjax import JaxArray, JaxRealArray, Shape
from tjax.dataclasses import field
from typing_extensions import override

__all__ = ['Support', 'ScalarSupport', 'VectorSupport', 'SimplexSupport', 'SymmetricMatrixSupport',
           'SquareMatrixSupport', 'Ring', 'RealField', 'ComplexField', 'BooleanRing',
           'IntegralRing']


class Ring:
    @abstractmethod
    def num_elements(self, support_num_element: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def flattened(self, x: JaxArray) -> JaxRealArray:
        raise NotImplementedError

    @abstractmethod
    def unflattened(self, y: JaxRealArray) -> JaxArray:
        raise NotImplementedError


class RealField(Ring):
    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        return x

    @override
    def unflattened(self, y: JaxRealArray) -> JaxArray:
        return y


class ComplexField(Ring):
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


class BooleanRing(Ring):
    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        return jnp.asarray(x, dtype=canonicalize_dtype(float))

    @override
    def unflattened(self, y: JaxRealArray) -> JaxArray:
        return jnp.asarray(y, dtype=jnp.bool_)


class IntegralRing(Ring):
    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        return jnp.asarray(x, dtype=canonicalize_dtype(float))

    @override
    def unflattened(self, y: JaxRealArray) -> JaxArray:
        return jnp.asarray(y, dtype=canonicalize_dtype(int))


real_field = RealField()
integral_ring = IntegralRing()
complex_field = ComplexField()
boolean_ring = BooleanRing()


class Support:
    @override
    def __init__(self, *, ring: Ring = real_field):
        super().__init__()
        self.ring = ring

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
        return self.ring.num_elements(1)

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        return self.ring.flattened(jnp.reshape(x, (*x.shape, 1)))

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int) -> JaxArray:
        x = self.ring.unflattened(y)
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
        return self.ring.num_elements(dimensions)

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        return self.ring.flattened(x)

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int) -> JaxArray:
        x = self.ring.unflattened(y)
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
        return self.ring.num_elements(dimensions - 1)

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        return self.ring.flattened(x)

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int) -> JaxArray:
        x = self.ring.unflattened(y)
        assert x.shape[-1] == dimensions
        return x


class SymmetricMatrixSupport(Support):
    @override
    def __init__(self, *, hermitian: bool = False, **kwargs: Any):
        if hermitian:
            kwargs.setdefault('ring', complex_field)
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
        return self.ring.num_elements(comb(dimensions + 1, 2))

    @override
    def flattened(self, x: JaxArray) -> JaxRealArray:
        dimensions = x.shape[-1]
        assert x.shape[-2] == dimensions
        index = (..., *jnp.triu_indices(dimensions))
        return self.ring.flattened(x[index])

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int) -> JaxArray:
        x = self.ring.unflattened(y)
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
        return self.ring.num_elements(dimensions ** 2)

    @override
    def flattened(self, x: JaxArray) -> JaxArray:
        y = jnp.reshape(x, (*x.shape[:-2], -1))
        return self.ring.flattened(y)

    @override
    def unflattened(self, y: JaxArray, dimensions: int) -> JaxArray:
        x = self.ring.unflattened(y)
        return jnp.reshape(x, x.shape[:-1] + self.shape(dimensions))


def distribution_parameter(support: Support,
                           *,
                           fixed: bool = False,
                           static: bool = False) -> Any:
    if static and not fixed:
        raise ValueError
    return field(static=static, metadata={'support': support, 'fixed': fixed})
