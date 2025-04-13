from __future__ import annotations

from abc import abstractmethod
from math import comb, isqrt
from typing import Any, cast

import array_api_extra as xpx
import jax.scipy.special as jss
import numpy as np
from array_api_compat import array_namespace
from numpy.random import Generator
from tjax import JaxArray, JaxRealArray, Shape
from typing_extensions import override

from ..types import Namespace
from .ring import ComplexField, RealField, Ring, complex_field, real_field


def triangular_number(k: int, /) -> int:
    return comb(k + 1, 2)


def triangular_number_index(k: int, /) -> int | None:
    discriminant = 1 + 8 * k
    sqrt_discriminant = isqrt(discriminant)
    if discriminant != sqrt_discriminant ** 2:
        return None
    assert sqrt_discriminant % 2 == 1
    return (sqrt_discriminant - 1) // 2


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
        return self.ring.num_elements(triangular_number(dimensions))

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
        deduced_dimensions = triangular_number_index(x.shape[-1])
        if deduced_dimensions is None:
            msg = f"The final dimension of the flattened vector, {x.shape[-1]}, is not triangular"
            raise ValueError(msg)
        if deduced_dimensions != dimensions:
            msg = (f"Deduced dimensions {deduced_dimensions} does not match provided dimensions"
                   f"{dimensions}.")
            raise ValueError(msg)
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
