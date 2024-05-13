from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from dataclasses import fields
from functools import reduce
from itertools import starmap
from typing import TYPE_CHECKING, Any, TypeVar

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from tjax import JaxComplexArray, JaxRealArray

from .iteration import parameters
from .parametrization import Parametrization
from .structure import Structure

Axis = int | tuple[int, ...]


def parameter_dot_product(x: NaturalParametrization[Any, Any], y: Any, /) -> JaxRealArray:
    """Return the vectorized dot product over all of the variable parameters."""
    def dotted_fields() -> Iterable[JaxRealArray]:
        xs = parameters(x, fixed=False, support=True).values()
        ys = parameters(y, fixed=False, support=True).values()

        for (x_value, x_support), (y_value, y_support) in zip(xs, ys, strict=True):
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
    paths = parameters(x, fixed=False).keys()
    iterators = [parameters(y, fixed=False).values()  # type: ignore[call-overload]
                 for y in (x, *ys)]
    final_values = list(starmap(operation, zip(*iterators, strict=True)))
    operated_fields = dict(zip(paths, final_values, strict=True))

    fixed_parameters = parameters(x, fixed=True)
    return Structure.create(x).assemble({**fixed_parameters, **operated_fields})


iv_ratio = tfp.math.bessel_iv_ratio
log_ive = tfp.math.log_bessel_ive


# Private functions --------------------------------------------------------------------------------
def _parameter_dot_product(x: JaxComplexArray, y: JaxComplexArray, n_axes: int) -> JaxRealArray:
    """Returns the real component of the dot product of the final n_axes axes of two arrays."""
    axes = tuple(range(-n_axes, 0))
    return jnp.sum(x * y, axis=axes).real


if TYPE_CHECKING:
    from .natural_parametrization import NaturalParametrization
