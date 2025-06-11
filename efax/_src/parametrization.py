from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from types import EllipsisType, ModuleType
from typing import Self, override

from array_api_compat import array_namespace
from tjax import JaxAbstractClass, JaxArray, Shape
from tjax.dataclasses import dataclass

from .parameter import Support


@dataclass
class Distribution(JaxAbstractClass):
    """The Distribution is the base class of all distributions."""
    def __getitem__(self, key: tuple[int | slice | EllipsisType | None, ...]) -> Self:
        from .iteration import parameters  # noqa: PLC0415
        from .structure.structure import Structure  # noqa: PLC0415
        parameters_ = {path: value[key] for path, value in parameters(self).items()}
        return Structure.create(self).assemble(parameters_)

    @abstractmethod
    def sub_distributions(self) -> Mapping[str, Distribution]:
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> Shape:
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @classmethod
    def adjust_support(cls, support: Support, name: str, **kwargs: JaxArray) -> Support:
        return support

    def __array_namespace__(self, api_version: str | None = None) -> ModuleType:  # noqa: PLW3201
        from .iteration import parameters  # noqa: PLC0415
        values = parameters(self).values()
        return array_namespace(*values)


@dataclass
class SimpleDistribution(Distribution):
    """A SimpleDistribution has no sub-distributions.

    As a consequence, its domain is a simple support (rather than a dict).
    """
    @override
    def sub_distributions(self) -> Mapping[str, Distribution]:
        return {}

    @classmethod
    @abstractmethod
    def domain_support(cls) -> Support:
        raise NotImplementedError
