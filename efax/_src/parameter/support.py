from __future__ import annotations

from abc import abstractmethod
from math import comb, isqrt
from typing import cast, override

import array_api_extra as xpx
import jax.scipy.special as jss
import numpy as np
from array_api_compat import array_namespace
from numpy.random import Generator
from opt_einsum import contract
from tjax import JaxArray, JaxRealArray, Shape, divide_where

from efax._src.types import Namespace

from .ring import ComplexField, RealField, Ring, complex_field, real_field


def triangular_number(k: int, /) -> int:
    return comb(k + 1, 2)


def triangular_number_index(k: int, /) -> int | None:
    discriminant = 1 + 8 * k
    sqrt_discriminant = isqrt(discriminant)
    if discriminant != sqrt_discriminant**2:
        return None
    assert sqrt_discriminant % 2 == 1
    return (sqrt_discriminant - 1) // 2


class Support:
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
    def generate(
        self, xp: Namespace, rng: Generator, shape: Shape, safety: float, dimensions: int
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
    def generate(
        self, xp: Namespace, rng: Generator, shape: Shape, safety: float, dimensions: int
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
    def generate(
        self, xp: Namespace, rng: Generator, shape: Shape, safety: float, dimensions: int
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
        if not map_to_plane:
            return self.ring.flattened(x, map_to_plane=False)
        xp = array_namespace(x)
        residual = 1.0 - xp.sum(x, axis=-1, keepdims=True)
        return xp.log(x / residual)

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        if not map_from_plane:
            x = self.ring.unflattened(y, map_from_plane=False)
            assert x.shape[-1] == dimensions - 1
            return x
        xp = array_namespace(y)
        assert y.shape[-1] == dimensions - 1
        logits = xp.concat((y, xp.zeros((*y.shape[:-1], 1), dtype=y.dtype)), axis=-1)
        return jss.softmax(logits, axis=-1)[..., :-1]

    @override
    def clamp(self, x: JaxArray) -> JaxArray:
        xp = array_namespace(x)
        eps = xp.finfo(x.dtype).eps
        s = xp.sum(x, axis=-1, keepdims=True)
        x *= xp.minimum(1.0, (1.0 - eps) / s)
        return xp.clip(x, min=eps, max=1.0 - eps)

    @override
    def generate(
        self, xp: Namespace, rng: Generator, shape: Shape, safety: float, dimensions: int
    ) -> JaxRealArray:
        alpha = xp.ones(dimensions)
        x = xp.asarray(rng.dirichlet(alpha, size=shape))
        if safety > 0.0:
            x = x * (1.0 - safety) + safety / dimensions
        return x[..., :-1]


class SubsimplexSupport(Support):
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
        if not map_to_plane:
            return self.ring.flattened(x, map_to_plane=False)
        xp = array_namespace(x)
        residual = 1.0 - xp.sum(x, axis=-1, keepdims=True)
        return xp.log(x / residual)

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        if not map_from_plane:
            x = self.ring.unflattened(y, map_from_plane=False)
            assert x.shape[-1] == dimensions
            return x
        xp = array_namespace(y)
        assert y.shape[-1] == dimensions
        logits = xp.concat((y, xp.zeros((*y.shape[:-1], 1), dtype=y.dtype)), axis=-1)
        return jss.softmax(logits, axis=-1)[..., :-1]

    @override
    def clamp(self, x: JaxArray) -> JaxArray:
        xp = array_namespace(x)
        eps = xp.finfo(x.dtype).eps
        s = xp.sum(x, axis=-1, keepdims=True)
        x *= xp.minimum(1.0, (1.0 - eps) / s)
        return xp.clip(x, min=eps, max=1.0 - eps)

    @override
    def generate(
        self, xp: Namespace, rng: Generator, shape: Shape, safety: float, dimensions: int
    ) -> JaxRealArray:
        alpha = xp.ones(dimensions + 1)
        x = xp.asarray(rng.dirichlet(alpha, size=shape))[..., :-1]
        if safety > 0.0:
            x *= 1.0 - safety
        return x


class _SelfAdjointMatrixSupport(Support):
    def __init__(
        self,
        *,
        positive_semidefinite: bool = False,
        negative_semidefinite: bool = False,
        ring: Ring = real_field,
    ) -> None:
        super().__init__(ring=ring)
        self.positive_semidefinite = positive_semidefinite
        self.negative_semidefinite = negative_semidefinite

    @override
    def axes(self) -> int:
        return 2

    @override
    def shape(self, dimensions: int) -> Shape:
        return (dimensions, dimensions)

    @override
    def generate(
        self, xp: Namespace, rng: Generator, shape: Shape, safety: float, dimensions: int
    ) -> JaxRealArray:
        # Generate a random matrix.
        m = self.ring.generate(xp, rng, (*shape, dimensions, dimensions), safety)
        if self.negative_semidefinite or self.positive_semidefinite:
            # Perform QR decomposition to obtain an orthogonal or unitary matrix Q.
            q, _ = xp.linalg.qr(m)
            # Generate Eigenvalues.
            assert isinstance(self.ring, RealField | ComplexField)
            eig_field = (
                RealField(minimum=0.0) if self.positive_semidefinite else RealField(maximum=0.0)
            )
            eig = eig_field.generate(xp, rng, (*shape, dimensions), safety)
            return contract("...ji,...j,...jk->...ik", self._left_factor(q), eig, q)
        return m + self._mirror(m)

    def _left_factor(self, x: JaxArray) -> JaxArray:
        return x

    def _mirror(self, x: JaxArray) -> JaxArray:
        return array_namespace(x).matrix_transpose(x)


class SymmetricMatrixSupport(_SelfAdjointMatrixSupport):
    @override
    def num_elements(self, dimensions: int) -> int:
        return self.ring.num_elements(triangular_number(dimensions))

    @override
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        xp = array_namespace(x)
        dimensions = x.shape[-1]
        assert x.shape[-2] == dimensions
        index_a, index_b = np.triu_indices(dimensions)
        x_triangular = xp.stack(
            [x[..., i, j] for i, j in zip(index_a, index_b, strict=True)], axis=-1
        )
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
            msg = (
                f"Deduced dimensions {deduced_dimensions} does not match provided dimensions"
                f"{dimensions}."
            )
            raise ValueError(msg)
        index_a, index_b = np.triu_indices(dimensions)
        result = xp.empty((*x.shape[:-1], dimensions, dimensions), dtype=x.dtype)
        for k, (i_, j_) in enumerate(zip(index_a, index_b, strict=True)):
            i = int(i_)
            j = int(j_)
            xk = x[..., k]
            result = xpx.at(result)[..., i, j].set(xk)  # ty: ignore
            if i != j:
                result = xpx.at(result)[..., j, i].set(xk)  # ty: ignore
        assert isinstance(result, JaxArray)
        return result


class HermitianMatrixSupport(_SelfAdjointMatrixSupport):
    def __init__(
        self,
        *,
        positive_semidefinite: bool = False,
        negative_semidefinite: bool = False,
    ) -> None:
        super().__init__(
            positive_semidefinite=positive_semidefinite,
            negative_semidefinite=negative_semidefinite,
            ring=complex_field,
        )

    @override
    def num_elements(self, dimensions: int) -> int:
        return dimensions**2

    @override
    def flattened(self, x: JaxArray, *, map_to_plane: bool) -> JaxRealArray:
        xp = array_namespace(x)
        dimensions = x.shape[-1]
        assert x.shape[-2] == dimensions
        index_a, index_b = np.triu_indices(dimensions)
        x_triangular = xp.stack(
            [x[..., i, j] for i, j in zip(index_a, index_b, strict=True)], axis=-1
        )
        off_diagonal = index_a != index_b
        return xp.concat(
            [
                xp.real(x_triangular),
                xp.imag(x_triangular[..., off_diagonal]),
            ],
            axis=-1,
        )

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        xp = array_namespace(y)
        if y.shape[-1] != dimensions**2:
            msg = (
                f"The final dimension of the flattened vector, {y.shape[-1]}, "
                f"is not {dimensions**2}"
            )
            raise ValueError(msg)
        triangular = triangular_number(dimensions)
        index_a, index_b = np.triu_indices(dimensions)
        off_diagonal = index_a != index_b
        imaginary = xp.zeros_like(y[..., :triangular])
        imaginary = xpx.at(imaginary)[..., off_diagonal].set(y[..., triangular:])  # ty: ignore
        x = y[..., :triangular] + 1j * imaginary  # ty: ignore
        result = xp.empty((*x.shape[:-1], dimensions, dimensions), dtype=x.dtype)
        for k, (i_, j_) in enumerate(zip(index_a, index_b, strict=True)):
            i = int(i_)
            j = int(j_)
            xk = x[..., k]
            result = xpx.at(result)[..., i, j].set(xk)
            if i != j:
                result = xpx.at(result)[..., j, i].set(xp.conj(xk))
        assert isinstance(result, JaxArray)
        return result

    @override
    def _left_factor(self, x: JaxArray) -> JaxArray:
        return array_namespace(x).conj(x)

    @override
    def _mirror(self, x: JaxArray) -> JaxArray:
        xp = array_namespace(x)
        return xp.conj(xp.matrix_transpose(x))


class SquareMatrixSupport(Support):
    @override
    def axes(self) -> int:
        return 2

    @override
    def shape(self, dimensions: int) -> Shape:
        return (dimensions, dimensions)

    @override
    def num_elements(self, dimensions: int) -> int:
        return self.ring.num_elements(dimensions**2)

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

    @override
    def generate(
        self, xp: Namespace, rng: Generator, shape: Shape, safety: float, dimensions: int
    ) -> JaxRealArray:
        return self.ring.generate(xp, rng, (*shape, dimensions, dimensions), safety)


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
        return x * divide_where(
            corrected_magnitude,
            magnitude,
            where=magnitude != 0.0,
            otherwise=xp.zeros_like(magnitude),
        )

    @override
    def unflattened(self, y: JaxRealArray, dimensions: int, *, map_from_plane: bool) -> JaxArray:
        if not map_from_plane:
            return y
        # y is in the plane.  Map it to the disk of the given radius.
        xp = array_namespace(y)
        assert y.shape[-1] == dimensions
        corrected_magnitude = cast("JaxRealArray", xp.linalg.norm(y, 2, axis=-1, keepdims=True))
        magnitude = self.radius * (jss.expit(corrected_magnitude) - 0.5) * 2.0
        return y * divide_where(
            magnitude,
            corrected_magnitude,
            where=corrected_magnitude != 0.0,
            otherwise=xp.zeros_like(corrected_magnitude),
        )
