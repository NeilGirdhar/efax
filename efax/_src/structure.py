from collections.abc import Callable, Iterable, Mapping
from functools import partial, reduce
from typing import Any, Generic, Self, TypeAlias, TypeVar, cast

import jax.numpy as jnp
from tjax import JaxComplexArray, JaxRealArray
from tjax.dataclasses import dataclass, field

from .iteration import parameters, support
from .parameter import Support
from .parametrization import Parametrization

Path: TypeAlias = tuple[str, ...]
Parameters: TypeAlias = Mapping[str, 'JaxComplexArray | Parameters']
Supports: TypeAlias = Mapping[str, 'Support | Supports']
ParametersAndSupports: TypeAlias = Mapping[
        str, 'tuple[JaxComplexArray, Support] | ParametersAndSupports']


@dataclass
class SubDistributionInfo:
    """A hashable collection of the static information for recreating sub-distributions."""
    path: Path
    type_: type[Parametrization]
    dimensions: int
    sub_distribution_names: list[str]


T = TypeVar('T')
P = TypeVar('P', bound=Parametrization)


@dataclass
class Structure(Generic[P]):
    """Divides a Parametrization into parameters and the information required to rebuid it.

    This is useful for operating on all the parameters.
    """
    # A post-order traversal of the tree.
    distributions: list[SubDistributionInfo] = field(static=True)

    @classmethod
    def create(cls, p: P) -> Self:
        return cls(cls._extract_distributions(p))

    def assemble(self, p: Mapping[Path, JaxComplexArray]) -> P:
        """Assemble a Parametrization from its parameters using the saved structure."""
        constructed: dict[Path, Parametrization | JaxComplexArray] = dict(p)
        for info in self.distributions:
            kwargs: dict[str, Parametrization | JaxComplexArray | dict[str, Any]] = {
                    name: constructed[*info.path, name]
                    for name in support(info.type_)}
            sub_distributions = {name: constructed[*info.path, name]
                                 for name in info.sub_distribution_names}
            if sub_distributions:
                kwargs['sub_distributions_objects'] = sub_distributions
            constructed[info.path] = info.type_(**kwargs)
        retval = constructed[()]
        assert isinstance(retval, Parametrization)
        return cast(P, retval)

    def reinterpret(self, q: Parametrization) -> P:
        """Reinterpret one parametrization using the saved structure."""
        p_paths = [(*info.path, name)
                   for info in self.distributions
                   for name in support(info.type_)]
        q_values = parameters(q, support=False).values()
        q_params_as_p = dict(zip(p_paths, q_values, strict=True))
        return self.assemble(q_params_as_p)

    @classmethod
    def _extract_distributions(cls, p: P) -> list[SubDistributionInfo]:
        return list(cls._walk(cls._make_info, p))

    @classmethod
    def _walk(cls,
              f: Callable[[Parametrization, Path], T],
              q: Parametrization,
              base_path: Path = (),
              ) -> Iterable[T]:
        """Post-order traversal of q."""
        for name, sub_distribution in q.sub_distributions().items():
            this_path = (*base_path, name)
            yield from cls._walk(f, sub_distribution, this_path)
        yield f(q, base_path)

    @classmethod
    def _make_info(cls, q: Parametrization, path: Path) -> SubDistributionInfo:
        from .interfaces.multidimensional import Multidimensional  # noqa: PLC0415
        dimensions = q.dimensions() if isinstance(q, Multidimensional) else 1
        sub_distribution_names = list(q.sub_distributions())
        return SubDistributionInfo(path, type(q), dimensions, sub_distribution_names)


@dataclass
class Flattener(Structure[P]):
    """Flattens a Parametrization into an array of variable parameters.

    Like Structure, it divides the parametrization---in this case, into the flattened parameters and
    the information required to rebuid the parametrization.

    This is useful when implementing machine learning algorithms since the variables parameters are
    the inputs and outputs of neural networks.
    """
    fixed_parameters: dict[Path, JaxComplexArray]

    def unflatten(self, flattened: JaxRealArray) -> P:
        consumed = 0
        constructed: dict[Path, Parametrization] = {}
        available = flattened.shape[-1]
        for info in self.distributions:
            kwargs: dict[str, Parametrization | JaxComplexArray | dict[str, Any]] = {}
            for name, this_support in support(info.type_, fixed=False).items():
                k = this_support.num_elements(info.dimensions)
                if consumed + k > available:
                    raise ValueError('Incompatible array')  # noqa: TRY003
                kwargs[name] = this_support.unflattened(flattened[..., consumed: consumed + k],
                                                        info.dimensions)
                consumed += k
            for name in support(info.type_, fixed=True):
                kwargs[name] = self.fixed_parameters[*info.path, name]
            sub_distributions = {name: constructed[*info.path, name]
                                 for name in info.sub_distribution_names}
            if sub_distributions:
                kwargs['sub_distributions_objects'] = sub_distributions
            constructed[info.path] = info.type_(**kwargs)
        if consumed != available:
            raise ValueError('Incompatible array')  # noqa: TRY003
        return cast(P, constructed[()])

    @classmethod
    def flatten(cls, p: P) -> tuple[Self, JaxRealArray]:
        arrays = [x
                  for xs in cls._walk(cls._make_flat, p)
                  for x in xs]
        flattened_array = reduce(partial(jnp.append, axis=-1), arrays)
        return (cls(cls._extract_distributions(p), parameters(p, fixed=True)),
                flattened_array)

    @classmethod
    def _make_flat(cls, q: Parametrization, path: Path) -> list[JaxRealArray]:
        return [support.flattened(value)
                for value, support in parameters(q, fixed=False, support=True,
                                                 recurse=False).values()]
