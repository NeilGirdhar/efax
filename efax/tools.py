from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any, Iterable, List

from chex import Array
from jax import numpy as jnp
from tjax import RealArray, field_values

from .parameter import parameter_names_values_axes

__all__: List[str] = []


def dot_final(x: Array, y: Array, n_axes: int) -> RealArray:
    """
    Returns the real component of the dot product of the final n_axes axes of two arrays.
    """
    axes = tuple(range(-n_axes, 0))
    return jnp.real(jnp.sum(x * y, axis=axes))


def tree_dot_final(x: NaturalParametrization[Any], y: Any) -> RealArray:
    def dotted_fields() -> Iterable[Array]:
        for (_, xf, n_axes), yf in zip(parameter_names_values_axes(x),
                                       field_values(y, static=False)):
            yield dot_final(xf, yf, n_axes)
    return reduce(jnp.add, dotted_fields())


if TYPE_CHECKING:
    from .natural_parametrization import NaturalParametrization
