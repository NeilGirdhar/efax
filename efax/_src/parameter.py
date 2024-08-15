from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from math import comb, sqrt
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
from jax.dtypes import canonicalize_dtype
from jax.experimental.array_api import matrix_transpose
from jax.scipy import special as jss
from numpy.random import Generator
from tjax import JaxArray, JaxComplexArray, JaxRealArray, Shape, inverse_softplus, softplus
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

    @abstractmethod
    def generate(self, rng: Generator, shape: Shape) -> JaxRealArray:
        raise NotImplementedError

    def clamp(self, x: JaxArray) -> JaxArray:
        return x


@dataclass
class RealField(Ring):
    minimum: float | None = None
    maximum: float | None = None
    generation_scale: float = 1.0  # Scale the generated random numbers to improve random testing.
    min_open: bool = True  # Open interval
    max_open: bool = True  # Open interval

    def __post_init__(self) -> None:
        dtype = canonicalize_dtype(float)
        eps = float(np.finfo(dtype).eps)
        if self.min_open and self.minimum is not None:
            self.minimum = float(max(self.minimum + eps,
                                     self.minimum * (1.0 + np.copysign(eps, self.minimum))))
        if self.max_open and self.maximum is not None:
            self.maximum = float(min(self.maximum - eps,
                                     self.maximum * (1.0 + np.copysign(eps, -self.maximum))))

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

    @override
    def generate(self, rng: Generator, shape: Shape) -> JaxRealArray:
        match self.minimum, self.maximum:
            case None, float(maximum):
                return jnp.asarray(maximum - rng.exponential(size=shape) * self.generation_scale)
            case float(minimum), None:
                return jnp.asarray(rng.exponential(size=shape) * self.generation_scale + minimum)
            case float(minimum), float(maximum):
                return jnp.asarray(minimum + (maximum - minimum) * rng.uniform(size=shape))
            case None, None:
                return jnp.asarray(rng.normal(scale=1.0, size=shape) * self.generation_scale)
            case _:
                raise TypeError

    @override
    def clamp(self, x: JaxRealArray) -> JaxRealArray:
        eps = jnp.finfo(x.dtype).eps
        minimum = None if self.minimum is None else self.minimum + eps
        maximum = None if self.maximum is None else self.maximum - eps
        return jnp.clip(x, min=minimum, max=maximum)


@dataclass
class ComplexField(Ring):
    minimum_modulus: float | JaxRealArray = 0.0
    maximum_modulus: None | float | JaxRealArray = None

    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element * 2

    @override
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        # if self.maximum_modulus is not None:
        #     assert False
        minimum = fix_bound(self.minimum_modulus, x)
        maximum = fix_bound(self.maximum_modulus, x)
        if map_to_plane:
            if maximum is None:
                # x is outside the disk of the given minimum.  Map it to the plane.
                magnitude = jnp.abs(x)
                corrected_magnitude = magnitude - minimum
                corrected_x = x * corrected_magnitude / magnitude
            else:
                # x is in the disk of the given maximum.  Map it to the plane.
                magnitude = jnp.abs(x)
                corrected_magnitude = jss.logit(((magnitude - minimum) / maximum) * 0.5 + 0.5)
                corrected_x = x * corrected_magnitude / magnitude
        else:
            corrected_x = x
        return jnp.concatenate([corrected_x.real, corrected_x.imag], axis=-1)

    @override
    def unflattened(self, y: JaxRealArray, *, map_from_plane: bool) -> JaxArray:
        assert y.shape[-1] % 2 == 0
        n = y.shape[-1] // 2
        corrected_x = y[..., :n] + 1j * y[..., n:]
        if not map_from_plane:
            return corrected_x
        return self._unflattened(corrected_x)

    def _unflattened(self, corrected_x: JaxComplexArray) -> JaxComplexArray:
        minimum = fix_bound(self.minimum_modulus, corrected_x)
        maximum = fix_bound(self.maximum_modulus, corrected_x)

        if maximum is None:
            # x is outside the disk of the given minimum.  Map it to the plane.
            corrected_magnitude = jnp.abs(corrected_x)
            magnitude = corrected_magnitude + minimum
            return corrected_x * magnitude / corrected_magnitude
        # x is in the disk of the given maximum.  Map it to the plane.
        corrected_magnitude = jnp.abs(corrected_x)
        magnitude = maximum * (jss.expit(corrected_magnitude) - 0.5) * 2.0 + minimum
        return corrected_x * magnitude / corrected_magnitude

    @override
    def generate(self, rng: Generator, shape: Shape) -> JaxRealArray:
        return self._unflattened(jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape)))


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

    @override
    def generate(self, rng: Generator, shape: Shape) -> JaxRealArray:
        return jnp.asarray(rng.binomial(1, 0.5, shape).astype(np.bool_))


@dataclass
class IntegralRing(Ring):
    minimum: int | None = None
    maximum: int | None = None

    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element

    @override
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        return jnp.asarray(x, dtype=canonicalize_dtype(float))

    @override
    def unflattened(self, y: JaxRealArray, *, map_from_plane: bool) -> JaxArray:
        return jnp.asarray(y, dtype=canonicalize_dtype(int))

    @override
    def generate(self, rng: Generator, shape: Shape) -> JaxRealArray:
        field = RealField(minimum=None if self.minimum is None else float(self.minimum),
                      maximum=None if self.maximum is None else float(self.maximum))
        real_values = field.generate(rng, shape) * 10
        return real_values.astype(canonicalize_dtype(int))


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

    @abstractmethod
    def generate(self, rng: Generator, shape: Shape, dimensions: int) -> JaxRealArray:
        raise NotImplementedError

    def clamp(self, x: JaxArray) -> JaxArray:
        return self.ring.clamp(x)


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
        return self.ring.flattened(x[..., jnp.newaxis], map_to_plane=map_to_plane)

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        x = self.ring.unflattened(y, map_from_plane=map_from_plane)
        assert x.shape[-1] == 1
        return x[..., 0]

    @override
    def generate(self, rng: Generator, shape: Shape, dimensions: int) -> JaxRealArray:
        return self.ring.generate(rng, shape)


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

    @override
    def generate(self, rng: Generator, shape: Shape, dimensions: int) -> JaxRealArray:
        return self.ring.generate(rng, (*shape, dimensions))


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

    @override
    def clamp(self, x: JaxArray) -> JaxArray:
        eps = jnp.finfo(x.dtype).eps
        s = jnp.sum(x, axis=-1, keepdims=True)
        x *= jnp.minimum(1.0, (1.0 - eps) / s)
        return jnp.clip(x, min=eps, max=1.0 - eps)

    @override
    def generate(self, rng: Generator, shape: Shape, dimensions: int) -> JaxRealArray:
        raise NotImplementedError


class SymmetricMatrixSupport(Support):
    @override
    def __init__(self,
                 *,
                 positive_semidefinite: bool = False,
                 negative_semidefinite: bool = False,
                 hermitian: bool = False,
                 **kwargs: Any):
        if hermitian:
            kwargs.setdefault('ring', complex_field)
        super().__init__(**kwargs)
        self.hermitian = hermitian
        self.positive_semidefinite = positive_semidefinite
        self.negative_semidefinite = negative_semidefinite

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

    @override
    def generate(self, rng: Generator, shape: Shape, dimensions: int) -> JaxRealArray:
        m = self.ring.generate(rng, (*shape, dimensions, dimensions))
        mt = matrix_transpose(m)
        if self.hermitian:
            mt = mt.conj()
        if self.negative_semidefinite or self.positive_semidefinite:
            assert isinstance(self.ring, RealField | ComplexField)
            eig_field = (RealField(minimum=0.0)
                         if self.positive_semidefinite else RealField(maximum=0.0))
            eig = eig_field.generate(rng, (*shape, dimensions))
            return jnp.einsum('...ij,...j,...jk->...ik', m, eig, mt)
        return m + mt


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
    return field(static=static, metadata={'support': support, 'fixed': fixed, 'parameter': True})


def fix_bound(x: JaxArray | float | None, y: JaxArray) -> None | JaxArray:
    if x is None:
        return x
    if isinstance(x, float):
        x = jnp.asarray(x)
    assert isinstance(x, JaxArray)
    while x.ndim < y.ndim:
        x = x[..., jnp.newaxis]
    return x
