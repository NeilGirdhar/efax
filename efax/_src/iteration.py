from collections.abc import Iterable, Mapping
from dataclasses import fields
from typing import Any, Literal, overload

from tjax import JaxComplexArray

from .parameter import Support
from .parametrization import Distribution, SimpleDistribution
from .types import Path


def flatten_mapping(m: Mapping[str, Any], /) -> dict[Path, Any]:
    """Flatten a nested mapping."""
    result: dict[Path, Any] = {}

    def _flatten(m: Mapping[str, Mapping[str, Any] | Any], prefix: Path) -> None:
        for key, value in m.items():
            path = (*prefix, key)
            if isinstance(value, Mapping):
                _flatten(value, path)
                continue
            result[path] = value
    _flatten(m, ())

    return result


def unflatten_mapping(m: Mapping[Path, Any], /) -> dict[str, Any]:
    """Unflatten a nested mapping."""
    result: dict[str, Any] = {}
    for path, value in m.items():
        cursor = result
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = value
    return result


# def parameters(p: Distribution,
#                     /,
#                     *,
#                     fixed: bool | None = None
#                     ) -> dict[str, dict[str, Any] | tuple[JaxComplexArray, Support]]:
#     """Walk the parameters of a distribution and return the variable ones.
#
#     Args:
#         p: The parametrization to walk.
#         fixed: If true or false, return the fixed or variable parameters, otherwise return both.
#
#     Returns:
#         The path, value, and support of each variable parameter.
#     """
#     return ({name: (value, support)
#              for name, value, support, is_fixed in p.parameters()
#              if fixed is None or is_fixed == fixed}
#             | {name: parameters(value, fixed=fixed)
#                 for name, value in p.sub_distributions()})


@overload
def parameters(p: Distribution,
               /,
               *,
               fixed: bool | None = None,
               support: Literal[False] = False,
               recurse: Literal[False],
               ) -> dict[str, JaxComplexArray]:
    ...


@overload
def parameters(p: Distribution,
               /,
               *,
               fixed: bool | None = None,
               support: Literal[True],
               recurse: Literal[False],
               ) -> dict[str, tuple[JaxComplexArray, Support]]:
    ...


@overload
def parameters(p: Distribution,
               /,
               *,
               fixed: bool | None = None,
               support: Literal[False] = False,
               recurse: Literal[True] = True,
               ) -> dict[Path, JaxComplexArray]:
    ...


@overload
def parameters(p: Distribution,
               /,
               *,
               fixed: bool | None = None,
               support: Literal[True],
               recurse: Literal[True] = True,
               ) -> dict[Path, tuple[JaxComplexArray, Support]]:
    ...


def parameters(p: Distribution,
               /,
               *,
               fixed: bool | None = None,
               support: bool = False,
               recurse: bool = True,
               ) -> Any:
    """Return the parameters of a distribution.

    Args:
        p: The parametrization to walk.
        fixed: If true or false, return the fixed or variable parameters, otherwise return both.
        support: If true, yield the support also.
        recurse: If true, recurse into sub-distributions.

    Returns:
        The path, value, and support of each variable parameter.
    """
    def _parameters(q: Distribution,
                    base_path: Path
                    ) -> Iterable[tuple[Any, ...]]:
        for this_field in fields(q):
            name = this_field.name
            value = getattr(q, name)
            metadata = this_field.metadata
            if not metadata.get('parameter', False):
                continue
            is_fixed = metadata['fixed']
            if not isinstance(is_fixed, bool):
                raise TypeError
            if fixed is not None and is_fixed != fixed:
                continue
            this_path = (*base_path, name) if recurse else name
            if support:
                yield this_path, value, metadata['support']
            else:
                yield this_path, value
        if not recurse:
            return
        for name, value in q.sub_distributions().items():
            this_path = (*base_path, name)
            yield from _parameters(value, this_path)
    if support:
        return {key: (value, support) for key, value, support in _parameters(p, ())}
    return dict(_parameters(p, ()))


def support(p: type[Distribution] | Distribution,
            /,
            *,
            fixed: bool | None = None,
            ) -> dict[str, Support]:
    """Return the support of a distribution class.

    Args:
        p: The parametrization to walk.
        fixed: If true or false, return the fixed or variable parameters, otherwise return both.

    Returns:
        The path, value, and support of each variable parameter.
    """
    def _parameters(q: type[Distribution],
                    base_path: Path
                    ) -> Iterable[tuple[str, Support]]:
        for this_field in fields(q):
            name = this_field.name
            metadata = this_field.metadata
            if not metadata.get('parameter', False):
                continue
            support = metadata['support']
            is_fixed = metadata['fixed']
            if not isinstance(is_fixed, bool):
                raise TypeError
            if fixed is not None and is_fixed != fixed:
                continue
            if not isinstance(support, Support):
                raise TypeError
            yield name, support
    cls_p: type[Distribution] = type(p) if isinstance(p, Distribution) else p
    return dict(_parameters(cls_p, ()))


def flat_dict_of_parameters(d: Distribution) -> dict[Path, SimpleDistribution]:
    from .transform.joint import JointDistribution  # noqa: PLC0415
    if isinstance(d, JointDistribution):
        return flatten_mapping(d.as_dict())
    assert isinstance(d, SimpleDistribution)
    return {(): d}


def flat_dict_of_observations(x: Mapping[str, Any] | JaxComplexArray
                              ) -> dict[Path, JaxComplexArray]:
    return flatten_mapping(x) if isinstance(x, Mapping) else {(): x}
