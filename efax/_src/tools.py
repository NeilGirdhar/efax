from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from dataclasses import fields
from functools import reduce
from typing import TYPE_CHECKING, Any, TypeVar

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from tjax import JaxComplexArray, JaxRealArray

from .parametrization import Parametrization

Axis = int | tuple[int, ...]


def parameter_dot_product(x: NaturalParametrization[Any, Any], y: Any, /) -> JaxRealArray:
    """Return the vectorized dot product over all of the variable parameters."""
    def dotted_fields() -> Iterable[JaxRealArray]:
        for (x_value, x_support), (y_value, y_support) in zip(x.parameters_value_support(),
                                                              y.parameters_value_support(),
                                                              strict=True):
            axes = x_support.axes()
            assert y_support.axes() == axes
            yield _parameter_dot_product(x_value, y_value, axes)
    return reduce(jnp.add, dotted_fields())


T = TypeVar('T', bound=Parametrization)


def parameter_mean(x: T, /, *, axis: Axis | None = None) -> T:
    """Return the mean of the parameters (fixed and variable)."""
    def mean_fields() -> Generator[tuple[str, JaxRealArray], None, None]:
        for field in fields(x):
            value = getattr(x, field.name)
            yield field.name, jnp.mean(value, axis=axis)

    return type(x)(**dict(mean_fields()))


def parameter_map(operation: Callable[..., JaxComplexArray],
                  x: T,
                  /,
                  *ys: Parametrization
                  ) -> T:
    """Return a new distribution created by operating on the variable fields of the inputs."""
    def operated_fields() -> Generator[tuple[str, JaxRealArray], None, None]:
        for name, x_value, _ in x.parameters_name_value_support():
            y_values = [getattr(yi, name) for yi in ys]
            yield name, operation(x_value, *y_values)

    fixed_parameters = x.fixed_parameters()
    return type(x)(**dict(operated_fields()), **fixed_parameters)


iv_ratio = tfp.math.bessel_iv_ratio
log_ive = tfp.math.log_bessel_ive


# Private functions --------------------------------------------------------------------------------
def _parameter_dot_product(x: JaxComplexArray, y: JaxComplexArray, n_axes: int) -> JaxRealArray:
    """Returns the real component of the dot product of the final n_axes axes of two arrays."""
    axes = tuple(range(-n_axes, 0))
    return jnp.sum(x * y, axis=axes).real


if TYPE_CHECKING:
    from .natural_parametrization import NaturalParametrization
