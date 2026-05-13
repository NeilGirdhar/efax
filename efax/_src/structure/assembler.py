from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Generic, cast

from numpy.random import Generator
from tjax import JaxComplexArray, JaxRealArray, Shape
from tjax.dataclasses import dataclass, field
from typing_extensions import TypeVar

from efax._src.analytic_continuation import ComplexContinuation
from efax._src.iteration import parameters
from efax._src.parameter import Support
from efax._src.parametrization import Distribution, SimpleDistribution
from efax._src.types import Namespace, Path

from .parameter_names import parameter_names
from .parameter_supports import parameter_supports

if TYPE_CHECKING:
    from efax._src.transform.joint import JointDistribution


@dataclass
class SimpleDistributionInfo:
    """Static information for recreating a simple distribution."""

    path: Path = field(static=True)
    type_: type[SimpleDistribution] = field(static=True)
    dimensions: int = field(static=True)


@dataclass
class JointDistributionInfo:
    """Static information for recreating a joint distribution."""

    path: Path = field(static=True)
    type_: type[JointDistribution] = field(static=True)
    dimensions: int = field(static=True)
    sub_distribution_names: list[str] = field(static=True)


type SubDistributionInfo = SimpleDistributionInfo | JointDistributionInfo

T = TypeVar("T")
P = TypeVar("P", bound=Distribution, default=Any)


@dataclass
class Assembler(Generic[P]):
    """Holds enough static information about a Distribution tree to reassemble one from raw data.

    Distribution classes alone lack the context needed to reconstruct themselves — for example, a
    joint distribution must know the concrete types of its sub-distributions at reconstruction time.
    Assembler captures that metadata (types, paths, dimensions) in a post-order traversal of the
    tree, enabling operations that would otherwise require a live instance:

    * Reassemble a Distribution from its parameters (assemble).
    * Coerce parameter values from one parametrization into another (coerce_from_distribution).
    * Enumerate domain support constraints (domain_support).
    * Generate a random distribution with valid parameters (generate_random).

    Assembler is the base class for Estimator, which extends it with parameter estimation,
    and Flattener, which extends Estimator with vectorization.
    """

    # A post-order traversal of the tree.
    infos: list[SubDistributionInfo] = field(static=True)

    @classmethod
    def create_assembler(cls, p: P) -> Assembler[P]:
        """Create an Assembler by extracting the distribution tree structure from p.

        Always returns a plain Assembler regardless of which subclass this is called on.
        Subclasses (Estimator, Flattener) use this to obtain an Assembler's infos before
        constructing their own richer objects.
        """
        return Assembler(cls._extract_distributions(p))

    def to_nat(self) -> Assembler:
        """Return a copy with distribution types converted to their natural parametrization."""
        from efax._src.expectation_parametrization import (  # noqa: PLC0415
            ExpectationParametrization,
        )

        infos = []
        for info in self.infos:
            if not issubclass(info.type_, ExpectationParametrization):
                msg = f"{info.type_.__name__} is not an EP"
                raise TypeError(msg)
            type_ = info.type_.natural_parametrization_cls()
            if isinstance(info, JointDistributionInfo):
                infos.append(
                    JointDistributionInfo(
                        info.path,
                        cast("type[JointDistribution]", type_),
                        info.dimensions,
                        info.sub_distribution_names,
                    )
                )
                continue
            infos.append(
                SimpleDistributionInfo(
                    info.path,
                    cast("type[SimpleDistribution]", type_),
                    info.dimensions,
                )
            )
        return Assembler(infos)

    def to_exp(self) -> Assembler:
        """Return a copy with distribution types converted to their expectation parametrization."""
        from efax._src.natural_parametrization import NaturalParametrization  # noqa: PLC0415

        infos: list[SubDistributionInfo] = []
        for info in self.infos:
            if not issubclass(info.type_, NaturalParametrization):
                msg = f"{info.type_.__name__} is not an NP"
                raise TypeError(msg)
            type_ = info.type_.expectation_parametrization_cls()
            if isinstance(info, JointDistributionInfo):
                infos.append(
                    JointDistributionInfo(
                        info.path,
                        cast("type[JointDistribution]", type_),
                        info.dimensions,
                        info.sub_distribution_names,
                    )
                )
                continue
            infos.append(
                SimpleDistributionInfo(
                    info.path,
                    cast("type[SimpleDistribution]", type_),
                    info.dimensions,
                )
            )
        return Assembler(infos)

    def assemble(self, params: Mapping[Path, JaxComplexArray | ComplexContinuation]) -> P:
        """Reassemble a Distribution from a mapping of paths to parameter arrays.

        This is the core operation: given the raw parameter values (keyed by their paths in the
        distribution tree), reconstruct the full Distribution object using the stored type metadata.
        """
        constructed: dict[Path, Distribution | JaxComplexArray | ComplexContinuation] = dict(params)
        for info in self.infos:
            if isinstance(info, JointDistributionInfo):
                sub_distributions = {
                    name: cast("Distribution", constructed[*info.path, name])
                    for name in info.sub_distribution_names
                }
                constructed[info.path] = info.type_(_sub_distributions=sub_distributions)
                continue
            kwargs: dict[
                str, Distribution | JaxComplexArray | ComplexContinuation | dict[str, Any]
            ] = {name: constructed[*info.path, name] for name in parameter_names(info.type_)}
            constructed[info.path] = info.type_(**kwargs)
        retval = constructed[()]
        assert isinstance(retval, Distribution)
        return cast("P", retval)

    def coerce_from_distribution(self, q: Distribution) -> P:
        """Coerce q's parameter values into this Assembler's distribution types.

        Takes the raw parameter arrays from q and pours them into the type schema stored in
        this Assembler, producing a distribution of type P with q's numeric values.  Useful
        for switching between parametrizations when the parameter layouts are known to match.
        """
        p_paths = [
            (*info.path, name) for info in self.infos for name in parameter_names(info.type_)
        ]
        q_values = parameters(q).values()
        q_params_as_p = dict(zip(p_paths, q_values, strict=True))
        return self.assemble(q_params_as_p)  # ty: ignore

    def domain_support(self) -> dict[Path, Support]:
        """Return the domain support constraints for each simple sub-distribution in the tree."""
        return {
            info.path: info.type_.domain_support()
            for info in self.infos
            if isinstance(info, SimpleDistributionInfo)
        }

    def generate_random(self, xp: Namespace, rng: Generator, shape: Shape, safety: float) -> P:
        """Generate a random distribution with parameters drawn from each parameter's support."""
        path_and_values: dict[tuple[str, ...], JaxRealArray] = {}
        for info in self.infos:
            if isinstance(info, JointDistributionInfo):
                continue
            for name, support, value_receptacle in parameter_supports(info.type_):
                value = support.generate(xp, rng, shape, safety, info.dimensions)
                path_and_values[*info.path, name] = value
                value_receptacle.set_value(value)
        return self.assemble(path_and_values)

    @classmethod
    def _extract_distributions(cls, p: P) -> list[SubDistributionInfo]:
        return list(cls._walk(cls._make_info, p))

    @classmethod
    def _walk(
        cls,
        f: Callable[[Distribution, Path], T],
        q: Distribution,
        base_path: Path = (),
    ) -> Iterable[T]:
        """Post-order traversal of q."""
        for name, sub_distribution in q.sub_distributions().items():
            this_path = (*base_path, name)
            yield from cls._walk(f, sub_distribution, this_path)
        yield f(q, base_path)

    @classmethod
    def _make_info(cls, q: Distribution, path: Path) -> SubDistributionInfo:
        from efax._src.interfaces.multidimensional import Multidimensional  # noqa: PLC0415
        from efax._src.transform.joint import JointDistribution  # noqa: PLC0415

        dimensions = q.dimensions() if isinstance(q, Multidimensional) else 1
        match q:
            case JointDistribution():
                return JointDistributionInfo(path, type(q), dimensions, list(q.sub_distributions()))
            case SimpleDistribution():
                return SimpleDistributionInfo(path, type(q), dimensions)
            case _:
                raise TypeError
