from collections.abc import Callable, Iterable, Mapping
from typing import Any, Generic, TypeVar, cast

from numpy.random import Generator
from tjax import JaxComplexArray, Shape
from tjax.dataclasses import dataclass, field

from ..iteration import parameters
from ..parameter import Support
from ..parametrization import Distribution, SimpleDistribution
from ..types import Namespace, Path
from .parameter_names import parameter_names
from .parameter_supports import parameter_supports


@dataclass
class SubDistributionInfo:
    """A hashable collection of the static information for recreating sub-distributions."""
    path: Path = field(static=True)
    type_: type[Distribution] = field(static=True)
    dimensions: int = field(static=True)
    sub_distribution_names: list[str] = field(static=True)


T = TypeVar('T')
P = TypeVar('P', bound=Distribution)
SP = TypeVar('SP', bound=SimpleDistribution)


@dataclass
class Structure(Generic[P]):
    """This class generalizes the notion of type for Distribution objects.

    Structure implements a variety of functions that would normally be class methods, but can't be
    because of missing information in a class.  Specifically:
    * Assemble a Distribution from its parameters using the saved structure.
    * Reinterpret a distribution in one parametrization using the saved structure.
    * Get the domain support, which may require the structure for joint distributions.
    * Generate a random set of parameters for a distribution having the saved structure.

    Structure is also the base class for Flattener.
    """
    # A post-order traversal of the tree.
    infos: list[SubDistributionInfo] = field(static=True)

    @classmethod
    def create(cls, p: P) -> 'Structure[P]':
        return Structure(cls._extract_distributions(p))

    def to_nat(self) -> 'Structure[Any]':
        from ..expectation_parametrization import ExpectationParametrization  # noqa: PLC0415
        infos = []
        for info in self.infos:
            assert issubclass(info.type_, ExpectationParametrization)
            infos.append(SubDistributionInfo(info.path, info.type_.natural_parametrization_cls(),
                                             info.dimensions, info.sub_distribution_names))
        return Structure(infos)

    def to_exp(self) -> 'Structure[Any]':
        from ..natural_parametrization import NaturalParametrization  # noqa: PLC0415
        infos = []
        for info in self.infos:
            assert issubclass(info.type_, NaturalParametrization)
            infos.append(SubDistributionInfo(info.path,
                                             info.type_.expectation_parametrization_cls(),
                                             info.dimensions, info.sub_distribution_names))
        return Structure(infos)

    def assemble(self, p: Mapping[Path, JaxComplexArray]) -> P:
        """Assemble a Distribution from its parameters using the saved structure."""
        constructed: dict[Path, Distribution | JaxComplexArray] = dict(p)
        for info in self.infos:
            kwargs: dict[str, Distribution | JaxComplexArray | dict[str, Any]] = {
                    name: constructed[*info.path, name]
                    for name in parameter_names(info.type_)}
            sub_distributions = {name: constructed[*info.path, name]
                                 for name in info.sub_distribution_names}
            if sub_distributions:
                kwargs['_sub_distributions'] = sub_distributions
            constructed[info.path] = info.type_(**kwargs)
        retval = constructed[()]
        assert isinstance(retval, Distribution)
        return cast('P', retval)

    def reinterpret(self, q: Distribution) -> P:
        """Reinterpret a distribution in one parametrization using the saved structure."""
        p_paths = [(*info.path, name)
                   for info in self.infos
                   for name in parameter_names(info.type_)]
        q_values = parameters(q).values()
        q_params_as_p = dict(zip(p_paths, q_values, strict=True))
        return self.assemble(q_params_as_p)

    def domain_support(self) -> dict[Path, Support]:
        """Get the domain support, which may require the structure for joint distributions."""
        return {info.path: info.type_.domain_support()
                for info in self.infos
                if issubclass(info.type_, SimpleDistribution)}

    def generate_random(self, xp: Namespace, rng: Generator, shape: Shape, safety: float) -> P:
        """Generate a random distribution."""
        path_and_values = {}
        for info in self.infos:
            for name, support, value_receptacle in parameter_supports(info.type_):
                value = support.generate(xp, rng, shape, safety, info.dimensions)
                path_and_values[*info.path, name] = value
                value_receptacle.set_value(value)
        return self.assemble(path_and_values)

    @classmethod
    def _extract_distributions(cls, p: P) -> list[SubDistributionInfo]:
        return list(cls._walk(cls._make_info, p))

    @classmethod
    def _walk(cls,
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
        from ..interfaces.multidimensional import Multidimensional  # noqa: PLC0415
        dimensions = q.dimensions() if isinstance(q, Multidimensional) else 1
        sub_distribution_names = list(q.sub_distributions())
        return SubDistributionInfo(path, type(q), dimensions, sub_distribution_names)
