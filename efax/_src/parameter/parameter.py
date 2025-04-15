from __future__ import annotations

from typing import Any

from tjax.dataclasses import field

from .support import Support


def distribution_parameter(support: Support,
                           *,
                           fixed: bool = False,
                           static: bool = False
                           ) -> Any:
    if static and not fixed:
        raise ValueError
    return field(static=static, metadata={'support': support, 'fixed': fixed, 'parameter': True})
