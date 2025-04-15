from collections.abc import Generator, Iterable
from dataclasses import fields

from ..parametrization import Distribution
from ..types import Path


def parameter_names(p: type[Distribution] | Distribution,
                    /,
                    *,
                    fixed: bool | None = None,
                    ) -> Generator[str]:
    """Return the parameter names in a distribution.

    Args:
        p: The distribution to walk.
        fixed: If true or false, return the fixed or variable parameters, otherwise return both.

    Returns:
        The name of each parameter.
    """
    def _parameters(q: type[Distribution],
                    base_path: Path
                    ) -> Iterable[str]:
        for this_field in fields(q):
            name = this_field.name
            metadata = this_field.metadata
            if not metadata.get('parameter', False):
                continue
            is_fixed = metadata['fixed']
            if not isinstance(is_fixed, bool):
                raise TypeError
            if fixed is not None and is_fixed != fixed:
                continue
            yield name
    cls_p: type[Distribution] = type(p) if isinstance(p, Distribution) else p
    yield from _parameters(cls_p, ())
