from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from tjax import JaxAbstractClass, Shape
from tjax.dataclasses import dataclass
from typing_extensions import Self

from .parameter import Support


@dataclass
class Parametrization(JaxAbstractClass):
    def __getitem__(self, key: Any) -> Self:
        from .iteration import parameters  # noqa: PLC0415
        from .structure import Structure  # noqa: PLC0415
        parameters_ = {path: value[key] for path, value in parameters(self).items()}
        return Structure.create(self).assemble(parameters_)

    def sub_distributions(self) -> Mapping[str, Parametrization]:
        return {}

    @property
    @abstractmethod
    def shape(self) -> Shape:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def domain_support(cls) -> Support:
        raise NotImplementedError
