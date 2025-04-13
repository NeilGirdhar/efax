from collections.abc import Generator
from dataclasses import dataclass, fields

from tjax import JaxComplexArray

from ..parameter import Support
from ..parametrization import Distribution
from ..types import Path


@dataclass
class ValueReceptacle:
    name: str
    parameter_values: dict[str, JaxComplexArray]

    def set_value(self, value: JaxComplexArray, /) -> None:
        self.parameter_values[self.name] = value


def parameter_supports(p: type[Distribution] | Distribution,
                       /,
                       *,
                       fixed: bool | None = None,
                       adjust: bool = True
                       ) -> Generator[tuple[str, Support, ValueReceptacle]]:
    """Return the parameter supports in a distribution.

    Args:
        p: The distribution to walk.
        fixed: If true or false, return the fixed or variable parameters, otherwise return both.
        adjust: If true, the support takes into account other values, and values must be sent to the
            receptacle.

    Returns:
        The name, support, and a receptacle for each parameter.
    """
    def _parameters(q: type[Distribution],
                    base_path: Path
                    ) -> Generator[tuple[str, Support, ValueReceptacle]]:
        parameter_values: dict[str, JaxComplexArray] = {}
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
            if adjust:
                support = q.adjust_support(support, name, **parameter_values)
            yield name, support, ValueReceptacle(name, parameter_values)
    cls_p: type[Distribution] = type(p) if isinstance(p, Distribution) else p
    yield from _parameters(cls_p, ())
