from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from math import comb, sqrt
from typing import Any, cast

import jax.numpy as jnp
from jax.dtypes import canonicalize_dtype
from jax.scipy import special as jss
from tjax import JaxArray, JaxRealArray, Shape, inverse_softplus, softplus
from tjax.dataclasses import field
from typing_extensions import override


class Ring:
    @abstractmethod
    def num_elements(self, support_num_element: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        raise NotImplementedError

    @abstractmethod
    def unflattened(self, y: JaxRealArray, *, map_from_plane: bool) -> JaxArray:
        raise NotImplementedError


@dataclass
class RealField(Ring):
    minimum: float | None = None
    maximum: float | None = None

    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element

    @override
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        if not map_to_plane:
            return x
        match self.minimum, self.maximum:
            case None, float(maximum):
                return -inverse_softplus(maximum - x)
            case float(minimum), None:
                return inverse_softplus(x - minimum)
            case float(minimum), float(maximum):
                return jss.logit((x - minimum) / (maximum - minimum))
            case _:
                return x

    @override
    def unflattened(self, y: JaxRealArray, *, map_from_plane: bool) -> JaxArray:
        if not map_from_plane:
            return y
        match self.minimum, self.maximum:
            case None, float(maximum):
                return maximum - softplus(-y)
            case float(minimum), None:
                return softplus(y) + minimum
            case float(minimum), float(maximum):
                return minimum + jss.expit(y) * (maximum - minimum)
            case _:
                return y


class ComplexField(Ring):
    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element * 2

    @override
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        return jnp.concatenate([x.real, x.imag], axis=-1)

    @override
    def unflattened(self, y: JaxRealArray, *, map_from_plane: bool) -> JaxArray:
        assert y.shape[-1] % 2 == 0
        n = y.shape[-1] // 2
        return y[..., :n] + 1j * y[..., n:]


class BooleanRing(Ring):
    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element

    @override
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        return jnp.asarray(x, dtype=canonicalize_dtype(float))

    @override
    def unflattened(self, y: JaxRealArray, *, map_from_plane: bool) -> JaxArray:
        return jnp.asarray(y, dtype=jnp.bool_)


class IntegralRing(Ring):
    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element

    @override
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        return jnp.asarray(x, dtype=canonicalize_dtype(float))

    @override
    def unflattened(self, y: JaxRealArray, *, map_from_plane: bool) -> JaxArray:
        return jnp.asarray(y, dtype=canonicalize_dtype(int))


real_field = RealField()
negative_support = RealField(maximum=0.0)
positive_support = RealField(minimum=0.0)
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
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        raise NotImplementedError

    @abstractmethod
    def unflattened(self, y: JaxRealArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
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
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        return self.ring.flattened(jnp.reshape(x, (*x.shape, 1)), map_to_plane=map_to_plane)

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        x = self.ring.unflattened(y, map_from_plane=map_from_plane)
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
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        return self.ring.flattened(x, map_to_plane=map_to_plane)

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        x = self.ring.unflattened(y, map_from_plane=map_from_plane)
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
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        return self.ring.flattened(x, map_to_plane=map_to_plane)

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        x = self.ring.unflattened(y, map_from_plane=map_from_plane)
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
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        dimensions = x.shape[-1]
        assert x.shape[-2] == dimensions
        index = (..., *jnp.triu_indices(dimensions))
        return self.ring.flattened(x[index], map_to_plane=map_to_plane)

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        x = self.ring.unflattened(y, map_from_plane=map_from_plane)
        k = x.shape[-1]
        sqrt_discriminant = sqrt(1 + 8 * k)
        i_sqrt_discriminant = int(sqrt_discriminant)
        if i_sqrt_discriminant != sqrt_discriminant:
            msg = f"{k} {sqrt_discriminant}"
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
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxArray:
        y = jnp.reshape(x, (*x.shape[:-2], -1))
        return self.ring.flattened(y, map_to_plane=map_to_plane)

    @override
    def unflattened(self, y: JaxArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        x = self.ring.unflattened(y, map_from_plane=map_from_plane)
        return jnp.reshape(x, x.shape[:-1] + self.shape(dimensions))


class CircularBoundedSupport(VectorSupport):
    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius

    @override
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        if not map_to_plane:
            return x
        # x is in the disk of the given radius.  Map it to the plane.
        magnitude = jnp.linalg.norm(x, 2, axis=-1, keepdims=True)
        corrected_magnitude = jss.logit((magnitude / self.radius) * 0.5 + 0.5)
        return x * corrected_magnitude / magnitude

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        if not map_from_plane:
            return y
        # y is in the plane.  Map it to the disk of the given radius.
        assert y.shape[-1] == dimensions
        corrected_magnitude = cast(JaxRealArray, jnp.linalg.norm(y, 2, axis=-1, keepdims=True))
        magnitude = self.radius * (jss.expit(corrected_magnitude) - 0.5) * 2.0
        return y * magnitude / corrected_magnitude


def distribution_parameter(support: Support,
                           *,
                           fixed: bool = False,
                           static: bool = False) -> Any:
    if static and not fixed:
        raise ValueError
    return field(static=static, metadata={'support': support, 'fixed': fixed})
