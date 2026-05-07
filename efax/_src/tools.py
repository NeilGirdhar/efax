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


def _walk_parameters(x: Distribution, y: Distribution, *, hermitian: bool) -> JaxComplexArray:
    def dotted_fields() -> Iterable[JaxComplexArray]:
        xs = parameters(x, fixed=False, support=True).values()
        ys = parameters(y, fixed=False, support=True).values()

        for (x_value, x_support), (y_value, y_support) in zip(xs, ys, strict=True):
            axes = x_support.axes()
            assert y_support.axes() == axes
            yield _parameter_inner_product(x_value, y_value, axes, hermitian=hermitian)

    dotted_fields_list = list(dotted_fields())
    xp = array_namespace(*dotted_fields_list)
    return reduce(xp.add, dotted_fields_list)


@jit
def parameter_dot_product(x: Distribution, y: Distribution, /) -> JaxRealArray:
    """Return the vectorized Hermitian dot product over all of the variable parameters."""
    return _walk_parameters(x, y, hermitian=True)


@jit
def parameter_holomorphic_dot(x: Distribution, y: Distribution, /) -> JaxComplexArray:
    """Vectorized holomorphic dot product over the variable parameters.

    Returns ``Σ x · y`` over each variable parameter's support axes (no
    conjugation, no real-part).  For real-dtype inputs this is equal to
    :func:`parameter_dot_product`.  Required by :func:`log_normalizer`'s custom
    JVP on the analytic-continuation path used by :meth:`characteristic_function`,
    where the primal output is complex and JAX expects a complex tangent.
    """
    return _walk_parameters(x, y, hermitian=False)


T = TypeVar("T", bound=Distribution)


def parameter_mean[T: Distribution](x: T, /, *, axis: Axis | None = None) -> T:
    """Return the mean over axis of all parameters, including fixed ones.

    Fixed parameters are constant along the reduced axes (having been broadcast to the sample shape
    in sufficient_statistics), so their mean equals their value.  Integer-typed parameters remain
    integers after the mean because the Array API spec requires mean to preserve integer dtypes.
    """
    xp = array_namespace(x)
    assembler = Assembler.create_assembler(x)
    all_params = parameters(x)
    averaged = {path: xp.mean(value, axis=axis) for path, value in all_params.items()}
    return assembler.assemble(averaged)


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
def _parameter_inner_product(
    x: JaxComplexArray,
    y: JaxComplexArray,
    n_axes: int,
    *,
    hermitian: bool,
) -> JaxComplexArray:
    """Per-tensor inner product over the final ``n_axes`` axes.

    With ``hermitian=True`` returns ``Σ Re(x · conj(y))`` (real-valued); with
    ``hermitian=False`` returns ``Σ x · y`` (the holomorphic form, complex when
    inputs are complex).
    """
    xp = array_namespace(x, y)
    axes = tuple(range(-n_axes, 0))
    if hermitian:
        return xp.sum(xp.real(x * xp.conj(y)), axis=axes)
    return xp.sum(x * y, axis=axes)
