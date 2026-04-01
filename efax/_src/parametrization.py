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
    """The base class of all distributions.

    A Distribution is a JAX-compatible dataclass whose fields are parameter arrays.
    The generic indexing operator slices all parameter arrays simultaneously, enabling
    vectorised operations over batches of distributions.
    """

    def __getitem__(self, key: tuple[int | slice | EllipsisType | None, ...]) -> Self:
        from .iteration import parameters  # noqa: PLC0415
        from .structure.assembler import Assembler  # noqa: PLC0415

        parameters_ = {path: value[key] for path, value in parameters(self).items()}
        return Assembler.create_assembler(self).assemble(parameters_)

    @abstractmethod
    def sub_distributions(self) -> Mapping[str, Distribution]:
        """Return the named sub-distributions that make up this distribution.

        Returns an empty mapping for simple (non-joint) distributions.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> Shape:
        """The batch shape of this distribution — the shape of a scalar statistic."""
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        """The number of batch dimensions (len(self.shape))."""
        return len(self.shape)

    @classmethod
    def adjust_support(cls, support: Support, name: str, **kwargs: JaxArray) -> Support:
        """Optionally adjust the support of a parameter given sibling parameter values.

        The default implementation returns the support unchanged.  Subclasses override
        this when the valid range of one parameter depends on another (e.g. a scale
        parameter whose minimum depends on a fixed dimension count).
        """
        return support

    def __array_namespace__(self, api_version: str | None = None) -> ModuleType:  # noqa: PLW3201
        from .iteration import parameters  # noqa: PLC0415

        values = parameters(self).values()
        return array_namespace(*values)


@dataclass
class SimpleDistribution(Distribution):
    """A Distribution with no sub-distributions.

    As a consequence, its observation domain is described by a single Support object
    rather than a dict of supports.
    """

    @override
    def sub_distributions(self) -> Mapping[str, Distribution]:
        return {}

    @classmethod
    @abstractmethod
    def domain_support(cls) -> Support:
        """Return the support that constrains observations drawn from this distribution."""
        raise NotImplementedError
