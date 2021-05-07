from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any, Iterable, List

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from tjax import ComplexArray, RealArray, RealNumeric

__all__: List[str] = []


def parameters_dot_product(x: NaturalParametrization[Any], y: Any) -> RealArray:
    def dotted_fields() -> Iterable[ComplexArray]:
        for (x_value, x_support), (y_value, y_support) in zip(x.parameters_value_support(),
                                                              y.parameters_value_support()):
            axes = x_support.axes()
            assert y_support.axes() == axes
            yield _parameter_dot_product(x_value, y_value, axes)
    return reduce(jnp.add, dotted_fields())


def inverse_softplus(y: RealArray) -> RealArray:
    return jnp.where(y > 80.0,
                     y,
                     jnp.log(jnp.expm1(y)))


ive = tfp.math.bessel_ive
log_ive = tfp.math.log_bessel_ive


def iv(v: RealNumeric, z: RealNumeric) -> RealNumeric:
    return tfp.math.bessel_ive(v, z) / jnp.exp(-jnp.abs(z))


# Private functions --------------------------------------------------------------------------------
def _parameter_dot_product(x: ComplexArray, y: ComplexArray, n_axes: int) -> RealArray:
    """
    Returns the real component of the dot product of the final n_axes axes of two arrays.
    """
    axes = tuple(range(-n_axes, 0))
    return jnp.real(jnp.sum(x * y, axis=axes))


if TYPE_CHECKING:
    from .natural_parametrization import NaturalParametrization
