from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, override

from array_api_compat import get_namespace
from jax import Array


class _IndexUpdateHelper(Protocol):
    def __getitem__(self, index: Any) -> _IndexUpdateRef: ...


class _IndexUpdateRef(Protocol):
    def set(self, values: Any) -> Array: ...


@dataclass
class GenericIndexUpdateRef(_IndexUpdateRef):
    x: Any
    index: Any

    @override
    def set(self, values: Any) -> Array:
        xp = get_namespace(self.x)
        retval = xp.asarray(self.x, copy=True)
        retval[self.index] = values
        return retval


@dataclass
class GenericIndexUpdateHelper(_IndexUpdateHelper):
    x: Any

    @override
    def __getitem__(self, index: Any) -> GenericIndexUpdateRef:
        return GenericIndexUpdateRef(self.x, index)


def array_at(x: Any) -> _IndexUpdateHelper:
    if isinstance(x, Array):
        return x.at
    return GenericIndexUpdateHelper(x)
