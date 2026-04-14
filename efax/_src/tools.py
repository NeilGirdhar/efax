from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from functools import reduce
from itertools import starmap
from typing import TypeVar

from array_api_compat import array_namespace
from jax import jit
from tjax import JaxComplexArray, JaxRealArray

from .iteration import parameters
from .parametrization import Distribution
from .structure.assembler import Assembler
from .types import Axis


@jit
def parameter_dot_product(x: Distribution, y: Distribution, /) -> JaxRealArray:
    """Return the vectorized dot product over all of the variable parameters."""

    def dotted_fields() -> Iterable[JaxRealArray]:
        xs = parameters(x, fixed=False, support=True).values()
        ys = parameters(y, fixed=False, support=True).values()

        for (x_value, x_support), (y_value, y_support) in zip(xs, ys, strict=True):
            axes = x_support.axes()
            assert y_support.axes() == axes
            yield _parameter_dot_product(x_value, y_value, axes)

    dotted_fields_list = list(dotted_fields())
    xp = array_namespace(*dotted_fields_list)
    return reduce(xp.add, dotted_fields_list)


T = TypeVar("T", bound=Distribution)


def parameter_mean[T: Distribution](x: T, /, *, axis: Axis | None = None) -> T:
    """Return the mean of the variable parameters; fixed parameters are preserved unchanged."""
    xp = array_namespace(x)
    structure = Assembler.create_assembler(x)
    fixed = parameters(x, fixed=True)
    variable = parameters(x, fixed=False)
    averaged = {path: xp.mean(value, axis=axis) for path, value in variable.items()}
    return structure.assemble({**fixed, **averaged})


def parameter_map[T: Distribution](
    operation: Callable[..., JaxComplexArray], x: T, /, *ys: Distribution
) -> T:
    """Return a new distribution created by operating on the variable fields of the inputs."""
    paths = parameters(x, fixed=False).keys()
    iterators = [parameters(y, fixed=False).values() for y in (x, *ys)]
    final_values = list(starmap(operation, zip(*iterators, strict=True)))
    operated_fields = dict(zip(paths, final_values, strict=True))

    fixed_parameters = parameters(x, fixed=True)
    return Assembler.create_assembler(x).assemble({**fixed_parameters, **operated_fields})


def join_mappings[T, V](**field_to_map: Mapping[T, V]) -> dict[T, dict[str, V]]:
    """Joins multiple mappings together using their common keys.

    >>> user_scores = {'elliot': 50, 'claris': 60}
    >>> user_times = {'elliot': 30, 'claris': 40}
    >>> join_mappings(score=user_scores, time=user_times)
    {'elliot': {'score': 50, 'time': 30}, 'claris': {'score': 60, 'time': 40}}
    """
    retval = defaultdict[T, dict[str, V]](dict)
    for field_name, mapping in field_to_map.items():
        for key, value in mapping.items():
            retval[key][field_name] = value
    return dict(retval)


# Private functions --------------------------------------------------------------------------------
def _parameter_dot_product(x: JaxComplexArray, y: JaxComplexArray, n_axes: int) -> JaxRealArray:
    """Returns the real component of the dot product of the final n_axes axes of two arrays."""
    xp = array_namespace(x, y)
    axes = tuple(range(-n_axes, 0))
    return xp.sum(xp.real(x * xp.conj(y)), axis=axes)
