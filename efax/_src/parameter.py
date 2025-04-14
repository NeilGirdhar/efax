from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from math import comb, sqrt
from types import ModuleType
from typing import Any, cast

import array_api_extra as xpx
import jax.numpy as jnp
import jax.scipy.special as jss
import numpy as np
from array_api_compat import array_namespace
from numpy.random import Generator
from tjax import JaxArray, JaxComplexArray, JaxRealArray, Shape, inverse_softplus, softplus
from tjax.dataclasses import field
from typing_extensions import override

from .types import Namespace


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
    def generate(self, xp: Namespace, rng: Generator, shape: Shape, safety: float) -> JaxRealArray:
        raise NotImplementedError

    def clamp(self, x: JaxArray) -> JaxArray:
        return x


def general_array_namespace(x: JaxRealArray | float) -> ModuleType:
    if isinstance(x, float):
        return np
    return array_namespace(x)


@dataclass
class RealField(Ring):
    minimum: float | JaxRealArray | None = None
    maximum: float | JaxRealArray | None = None
    generation_scale: float = 1.0  # Scale the generated random numbers to improve random testing.
    min_open: bool = True  # Open interval
    max_open: bool = True  # Open interval

    def __post_init__(self) -> None:
        dtype = jnp.empty((), dtype=float).dtype  # This is canonicalize_dtype(float).
        eps = float(np.finfo(dtype).eps)
        if self.min_open and self.minimum is not None:
            xp = general_array_namespace(self.minimum)
            self.minimum = jnp.asarray(jnp.maximum(
                self.minimum + eps,
                self.minimum * (1.0 + xp.copysign(eps, self.minimum))))
        if self.max_open and self.maximum is not None:
            xp = general_array_namespace(self.maximum)
            self.maximum = jnp.asarray(jnp.minimum(
                self.maximum - eps,
                self.maximum * (1.0 + xp.copysign(eps, -self.maximum))))

    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element

    @override
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        xp = array_namespace(x)
        if not map_to_plane:
            return x
        if self.minimum is None and self.maximum is not None:
            maximum = xp.asarray(self.maximum)
            return -inverse_softplus(maximum - x)
        if self.minimum is not None and self.maximum is None:
            minimum = xp.asarray(self.minimum)
            return inverse_softplus(x - minimum)
        if self.minimum is not None and self.maximum is not None:
            minimum = xp.asarray(self.minimum)
            maximum = xp.asarray(self.maximum)
            return jss.logit((x - minimum) / (maximum - minimum))
        return x

    @override
    def unflattened(self, y: JaxRealArray, *, map_from_plane: bool) -> JaxArray:
        xp = array_namespace(y)
        if not map_from_plane:
            return y
        if self.minimum is None and self.maximum is not None:
            maximum = xp.asarray(self.maximum)
            return maximum - softplus(-y)
        if self.minimum is not None and self.maximum is None:
            minimum = xp.asarray(self.minimum)
            return softplus(y) + minimum
        if self.minimum is not None and self.maximum is not None:
            minimum = xp.asarray(self.minimum)
            maximum = xp.asarray(self.maximum)
            return minimum + jss.expit(y) * (maximum - minimum)
        return y

    @override
    def generate(self, xp: Namespace, rng: Generator, shape: Shape, safety: float) -> JaxRealArray:
        if self.minimum is None and self.maximum is not None:
            maximum = xp.asarray(self.maximum)
            return xp.asarray(maximum - rng.exponential(size=shape) * self.generation_scale
                              - safety)
        if self.minimum is not None and self.maximum is None:
            minimum = xp.asarray(self.minimum)
            return xp.asarray(rng.exponential(size=shape) * self.generation_scale + minimum
                              + safety)
        if self.minimum is not None and self.maximum is not None:
            minimum = xp.asarray(self.minimum)
            maximum = xp.asarray(self.maximum)
            domain_width = xp.maximum(maximum - minimum - safety, 0.0)
            true_minimum = xp.mean(minimum, maximum) - domain_width * 0.5
            return xp.asarray(true_minimum + domain_width * rng.uniform(size=shape))
        return xp.asarray(rng.normal(scale=1.0, size=shape) * self.generation_scale)

    @override
    def clamp(self, x: JaxRealArray) -> JaxRealArray:
        xp = array_namespace(x)
        eps = xp.finfo(x.dtype).eps
        minimum = None if self.minimum is None else self.minimum + eps
        maximum = None if self.maximum is None else self.maximum - eps
        return xp.clip(x, min=minimum, max=maximum)


@dataclass
class ComplexField(Ring):
    minimum_modulus: float | JaxRealArray = 0.0
    maximum_modulus: float | JaxRealArray | None = None

    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element * 2

    @override
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        xp = array_namespace(x)
        # if self.maximum_modulus is not None:
        #     assert False
        minimum = fix_bound(self.minimum_modulus, x)
        maximum = fix_bound(self.maximum_modulus, x)
        if map_to_plane:
            if maximum is None:
                # x is outside the disk of the given minimum.  Map it to the plane.
                magnitude = xp.abs(x)
                corrected_magnitude = magnitude - minimum
                corrected_x = x * corrected_magnitude / magnitude
            else:
                # x is in the disk of the given maximum.  Map it to the plane.
                magnitude = xp.abs(x)
                corrected_magnitude = jss.logit(((magnitude - minimum) / maximum) * 0.5 + 0.5)
                corrected_x = x * corrected_magnitude / magnitude
        else:
            corrected_x = x
        return xp.concat([xp.real(corrected_x), xp.imag(corrected_x)], axis=-1)

    @override
    def unflattened(self, y: JaxRealArray, *, map_from_plane: bool) -> JaxArray:
        assert y.shape[-1] % 2 == 0
        n = y.shape[-1] // 2
        corrected_x = y[..., :n] + 1j * y[..., n:]
        if not map_from_plane:
            return corrected_x
        return self._unflattened(corrected_x)

    def _unflattened(self, corrected_x: JaxComplexArray) -> JaxComplexArray:
        xp = array_namespace(corrected_x)
        minimum = fix_bound(self.minimum_modulus, corrected_x)
        maximum = fix_bound(self.maximum_modulus, corrected_x)

        if maximum is None:
            # x is outside the disk of the given minimum.  Map it to the plane.
            corrected_magnitude = xp.abs(corrected_x)
            magnitude = corrected_magnitude + minimum
            return corrected_x * magnitude / corrected_magnitude
        # x is in the disk of the given maximum.  Map it to the plane.
        corrected_magnitude = xp.abs(corrected_x)
        magnitude = maximum * (jss.expit(corrected_magnitude) - 0.5) * 2.0 + minimum
        return corrected_x * magnitude / corrected_magnitude

    @override
    def generate(self, xp: Namespace, rng: Generator, shape: Shape, safety: float) -> JaxRealArray:
        return self._unflattened(xp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape)))


class BooleanRing(Ring):
    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element

    @override
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        xp = array_namespace(x)
        return xp.asarray(x, dtype=float)

    @override
    def unflattened(self, y: JaxRealArray, *, map_from_plane: bool) -> JaxArray:
        xp = array_namespace(y)
        return xp.asarray(y, dtype=xp.bool_)

    @override
    def generate(self, xp: Namespace, rng: Generator, shape: Shape, safety: float) -> JaxRealArray:
        return xp.asarray(rng.binomial(1, 0.5, shape), dtype=xp.bool_)


@dataclass
class IntegralRing(Ring):
    minimum: int | None = None
    maximum: int | None = None

    @override
    def num_elements(self, support_num_element: int) -> int:
        return support_num_element

    @override
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        xp = array_namespace(x)
        return xp.asarray(x, dtype=float)

    @override
    def unflattened(self, y: JaxRealArray, *, map_from_plane: bool) -> JaxArray:
        xp = array_namespace(y)
        return xp.asarray(y, dtype=int)

    @override
    def generate(self, xp: Namespace, rng: Generator, shape: Shape, safety: float) -> JaxRealArray:
        field = RealField(minimum=None if self.minimum is None else float(self.minimum),
                      maximum=None if self.maximum is None else float(self.maximum))
        real_values = field.generate(xp, rng, shape, safety) * 10
        return xp.astype(real_values, int)


real_field = RealField()
negative_support = RealField(maximum=0.0)
positive_support = RealField(minimum=0.0)
integral_ring = IntegralRing()
complex_field = ComplexField()
boolean_ring = BooleanRing()


class Support:
    @override
    def __init__(self, *, ring: Ring = real_field) -> None:
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
    def generate(self, xp: Namespace, rng: Generator, shape: Shape, safety: float, dimensions: int
                 ) -> JaxRealArray:
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
        xp = array_namespace(x)
        return self.ring.flattened(x[..., xp.newaxis], map_to_plane=map_to_plane)

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        x = self.ring.unflattened(y, map_from_plane=map_from_plane)
        assert x.shape[-1] == 1
        return x[..., 0]

    @override
    def generate(self, xp: Namespace, rng: Generator, shape: Shape, safety: float, dimensions: int
                 ) -> JaxRealArray:
        return self.ring.generate(xp, rng, shape, safety)


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
    def generate(self, xp: Namespace, rng: Generator, shape: Shape, safety: float, dimensions: int
                 ) -> JaxRealArray:
        return self.ring.generate(xp, rng, (*shape, dimensions), safety)


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
        xp = array_namespace(x)
        eps = xp.finfo(x.dtype).eps
        s = xp.sum(x, axis=-1, keepdims=True)
        x *= xp.minimum(1.0, (1.0 - eps) / s)
        return xp.clip(x, min=eps, max=1.0 - eps)

    @override
    def generate(self, xp: Namespace, rng: Generator, shape: Shape, safety: float, dimensions: int
                 ) -> JaxRealArray:
        raise NotImplementedError


class SymmetricMatrixSupport(Support):
    @override
    def __init__(self,
                 *,
                 positive_semidefinite: bool = False,
                 negative_semidefinite: bool = False,
                 hermitian: bool = False,
                 **kwargs: Any) -> None:
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
        xp = array_namespace(x)
        dimensions = x.shape[-1]
        assert x.shape[-2] == dimensions
        index_a, index_b = np.triu_indices(dimensions)
        x_triangular = xp.stack([x[..., i, j]
                                 for i, j in zip(index_a, index_b, strict=True)],
                                axis=-1)
        return self.ring.flattened(x_triangular, map_to_plane=map_to_plane)

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        xp = array_namespace(y)
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
        index_a, index_b = np.triu_indices(dimensions)
        result = xp.empty((*x.shape[:-1], dimensions, dimensions), dtype=x.dtype)
        for k, (i_, j_) in enumerate(zip(index_a, index_b, strict=True)):
            i = int(i_)
            j = int(j_)
            xk = x[..., k]
            result = xpx.at(result)[..., i, j].set(xk)
            if i != j:
                cxk = xp.conj(xk) if self.hermitian else xk
                result = xpx.at(result)[..., j, i].set(cxk)
        assert isinstance(result, JaxArray)
        return result

    @override
    def generate(self, xp: Namespace, rng: Generator, shape: Shape, safety: float, dimensions: int
                 ) -> JaxRealArray:
        # Generate a random matrix.
        m = self.ring.generate(xp, rng, (*shape, dimensions, dimensions), safety)
        if self.negative_semidefinite or self.positive_semidefinite:
            # Perform QR decomposition to obtain an orthogonal matrix Q
            q, _ = np.linalg.qr(m)
            # Generate Eigenvalues.
            assert isinstance(self.ring, RealField | ComplexField)
            eig_field = (RealField(minimum=0.0)
                         if self.positive_semidefinite else RealField(maximum=0.0))
            eig = eig_field.generate(xp, rng, (*shape, dimensions), safety)
            # Return Q.T @ diag(eig) @ Q.
            return xp.einsum('...ji,...j,...jk->...ik', xp.conj(q) if self.hermitian else q, eig, q)
        mt = xp.matrix_transpose(m)
        if self.hermitian:
            mt = xp.conj(mt)
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
        xp = array_namespace(x)
        y = xp.reshape(x, (*x.shape[:-2], -1))
        return self.ring.flattened(y, map_to_plane=map_to_plane)

    @override
    def unflattened(self, y: JaxArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        xp = array_namespace(y)
        x = self.ring.unflattened(y, map_from_plane=map_from_plane)
        return xp.reshape(x, x.shape[:-1] + self.shape(dimensions))


class CircularBoundedSupport(VectorSupport):
    def __init__(self, radius: float) -> None:
        super().__init__()
        self.radius = radius

    @override
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        if not map_to_plane:
            return x
        # x is in the disk of the given radius.  Map it to the plane.
        xp = array_namespace(x)
        magnitude = xp.linalg.norm(x, 2, axis=-1, keepdims=True)
        corrected_magnitude = jss.logit((magnitude / self.radius) * 0.5 + 0.5)
        return x * corrected_magnitude / magnitude

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        if not map_from_plane:
            return y
        # y is in the plane.  Map it to the disk of the given radius.
        xp = array_namespace(y)
        assert y.shape[-1] == dimensions
        corrected_magnitude = cast('JaxRealArray', xp.linalg.norm(y, 2, axis=-1, keepdims=True))
        magnitude = self.radius * (jss.expit(corrected_magnitude) - 0.5) * 2.0
        return y * magnitude / corrected_magnitude


def distribution_parameter(support: Support,
                           *,
                           fixed: bool = False,
                           static: bool = False) -> Any:
    if static and not fixed:
        raise ValueError
    return field(static=static, metadata={'support': support, 'fixed': fixed, 'parameter': True})


def fix_bound(x: JaxArray | float | None, y: JaxArray) -> JaxArray | None:
    xp = array_namespace(y)
    if x is None:
        return x
    if isinstance(x, float):
        x = xp.asarray(x)
    assert isinstance(x, JaxArray)
    xp = array_namespace(x, y)
    while x.ndim < y.ndim:
        x = x[..., xp.newaxis]
    return x
