from __future__ import annotations

from collections.abc import Iterable
from functools import reduce
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from tjax import JaxComplexArray, JaxRealArray

__all__: list[str] = []


def parameters_dot_product(x: NaturalParametrization[Any, Any], y: Any) -> JaxRealArray:
    def dotted_fields() -> Iterable[JaxRealArray]:
        for (x_value, x_support), (y_value, y_support) in zip(x.parameters_value_support(),
                                                              y.parameters_value_support(),
                                                              strict=True):
            axes = x_support.axes()
            assert y_support.axes() == axes
            yield _parameter_dot_product(x_value, y_value, axes)
    return reduce(jnp.add, dotted_fields())


iv_ratio = tfp.math.bessel_iv_ratio
log_ive = tfp.math.log_bessel_ive
betaln = tfp.math.lbeta


# Private functions --------------------------------------------------------------------------------
def _parameter_dot_product(x: JaxComplexArray, y: JaxComplexArray, n_axes: int) -> JaxRealArray:
    """Returns the real component of the dot product of the final n_axes axes of two arrays."""
    axes = tuple(range(-n_axes, 0))
    return jnp.sum(x * y, axis=axes).real


if TYPE_CHECKING:
    from .natural_parametrization import NaturalParametrization
