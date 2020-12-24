from typing import Any, List

from tjax import field

__all__: List[str] = []


def distribution_parameter(axes: int) -> Any:
    return field(metadata={'axes': axes})
