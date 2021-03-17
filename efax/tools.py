from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any, Iterable, List

import jax.numpy as jnp
from chex import Array
from tensorflow_probability.substrates import jax as tfp
from tjax import RealArray, field_values

from .parameter import parameter_names_values_support

__all__: List[str] = []


def dot_final(x: Array, y: Array, n_axes: int) -> RealArray:
    """
    Returns the real component of the dot product of the final n_axes axes of two arrays.
    """
    axes = tuple(range(-n_axes, 0))
    return jnp.real(jnp.sum(x * y, axis=axes))


def tree_dot_final(x: NaturalParametrization[Any], y: Any) -> RealArray:
    def dotted_fields() -> Iterable[Array]:
        for (_, xf, support), yf in zip(parameter_names_values_support(x),
                                        field_values(y, static=False)):
            yield dot_final(xf, yf, support.axes())
    return reduce(jnp.add, dotted_fields())


def inverse_softplus(y: RealArray) -> RealArray:
    return jnp.where(y > 80.0,
                     y,
                     jnp.log(jnp.expm1(y)))


ive = tfp.math.bessel_ive


def iv(v: Array, z: Array) -> Array:
    return tfp.math.bessel_ive(v, z) / jnp.exp(-jnp.abs(z))


if TYPE_CHECKING:
    from .natural_parametrization import NaturalParametrization
