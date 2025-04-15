from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from types import ModuleType

import jax.numpy as jnp
import jax.scipy.special as jss
import numpy as np
from array_api_compat import array_namespace
from numpy.random import Generator
from tjax import JaxArray, JaxComplexArray, JaxRealArray, Shape, inverse_softplus, softplus
from typing_extensions import override

from ..types import Namespace


def _fix_bound(bound: JaxArray | float | None, x: JaxArray) -> JaxArray | None:
    xp = array_namespace(x)
    if bound is None:
        return bound
    bound_x = xp.asarray(bound)
    assert bound_x.shape == x.shape[:bound_x.ndim]
    delta_ndim = x.ndim - bound_x.ndim
    if delta_ndim == 0:
        return bound_x
    assert delta_ndim > 0
    return xp.reshape(bound_x, (*bound_x.shape, *((1,) * delta_ndim)))


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
        minimum = _fix_bound(self.minimum, x)
        maximum = _fix_bound(self.maximum, x)
        if not map_to_plane:
            return x
        if minimum is None and maximum is not None:
            return -inverse_softplus(maximum - x)
        if minimum is not None and maximum is None:
            return inverse_softplus(x - minimum)
        if minimum is not None and maximum is not None:
            return jss.logit((x - minimum) / (maximum - minimum))
        return x

    @override
    def unflattened(self, y: JaxRealArray, *, map_from_plane: bool) -> JaxArray:
        minimum = _fix_bound(self.minimum, y)
        maximum = _fix_bound(self.maximum, y)
        if not map_from_plane:
            return y
        if minimum is None and maximum is not None:
            return maximum - softplus(-y)
        if minimum is not None and maximum is None:
            return softplus(y) + minimum
        if minimum is not None and maximum is not None:
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
        minimum = _fix_bound(self.minimum_modulus, x)
        maximum = _fix_bound(self.maximum_modulus, x)
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
        minimum = _fix_bound(self.minimum_modulus, corrected_x)
        maximum = _fix_bound(self.maximum_modulus, corrected_x)

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
