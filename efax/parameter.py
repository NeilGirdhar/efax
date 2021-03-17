from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Tuple, Type, Union

from jax import numpy as jnp
from jax.nn import sigmoid, softplus
from tjax import Array, field, field_names_values_metadata, fields

__all__ = ['parameter_names_values_axes', 'parameter_names_axes_support', 'Support', 'RealSupport',
           'ComplexSupport', 'BoundedRealSupport', 'BoundedAboveRealSupport',
           'BoundedBelowRealSupport', 'real_support', 'complex_support', 'negative_support']


class Support:
    def soft_clip(self, x: Array) -> Array:
        raise NotImplementedError


class RealSupport(Support):
    def soft_clip(self, x: Array) -> Array:
        return x


class ComplexSupport(Support):
    def soft_clip(self, x: Array) -> Array:
        return x


class BoundedRealSupport(Support):
    def __init__(self, minimum: float, maximum: float):
        self.minimum = minimum
        self.maximum = maximum

    def soft_clip(self, x: Array) -> Array:
        return self.minimum + sigmoid(x) * (self.maximum - self.minimum)


class BoundedAboveRealSupport(Support):
    def __init__(self, maximum: float):
        self.maximum = maximum

    def soft_clip(self, x: Array) -> Array:
        return self.maximum - softplus(-x)


class BoundedBelowRealSupport(Support):
    def __init__(self, minimum: float):
        self.minimum = minimum

    def soft_clip(self, x: Array) -> Array:
        return softplus(x) + self.minimum


class CircularBoundedSupport(Support):
    def __init__(self, radius: float):
        self.radius = radius

    def soft_clip(self, x: Array) -> Array:
        kappa = jnp.linalg.norm(x, 2, axis=-1, keepdims=True)
        return jnp.where(kappa == 0.0, x, x * (self.radius * sigmoid(kappa) / kappa))
        return softplus(x) + self.minimum


real_support = RealSupport()
complex_support = ComplexSupport()
negative_support = BoundedAboveRealSupport(0.0)
positive_support = BoundedBelowRealSupport(0.0)


def distribution_parameter(support: Support, *, axes: int) -> Any:
    if not isinstance(axes, int):
        raise TypeError
    if not isinstance(support, Support):
        raise TypeError
    return field(metadata={'axes': axes, 'support': support})


def parameter_names_values_axes(x: Parametrization) -> Iterable[Tuple[str, Array, int]]:
    for name, value, metadata in field_names_values_metadata(x, static=False):
        n_axes = metadata['axes']
        yield name, value, n_axes


def parameter_names_axes_support(x: Union[Type[Parametrization],
                                          Parametrization]) -> Iterable[Tuple[str, int, Support]]:
    for this_field in fields(x, static=False):
        n_axes = this_field.metadata['axes']
        support = this_field.metadata['support']
        yield this_field.name, n_axes, support


if TYPE_CHECKING:
    from .expectation_parametrization import Parametrization
