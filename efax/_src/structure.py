from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from functools import partial, reduce
from typing import Any, Generic, Self, TypeVar, cast

import jax.numpy as jnp
from tjax import JaxComplexArray, JaxRealArray
from tjax.dataclasses import dataclass, field

from .iteration import parameters, support
from .parameter import Support
from .parametrization import GeneralParametrization, Parametrization
from .types import Path


@dataclass
class SubDistributionInfo:
    """A hashable collection of the static information for recreating sub-distributions."""
    path: Path
    type_: type[GeneralParametrization]
    dimensions: int
    sub_distribution_names: list[str]


T = TypeVar('T')
P = TypeVar('P', bound=GeneralParametrization)


@dataclass
class Structure(Generic[P]):
    """Structure is a generalization of type when it comes to Parametrization objects.

    Divides a GeneralParametrization into parameters and the information required to rebuid it.

    This is useful for operating on all the parameters.
    """
    # A post-order traversal of the tree.
    distributions: list[SubDistributionInfo] = field(static=True)

    @classmethod
    def create(cls, p: P) -> Self:
        return cls(cls._extract_distributions(p))

    def assemble(self, p: Mapping[Path, JaxComplexArray]) -> P:
        """Assemble a GeneralParametrization from its parameters using the saved structure."""
        constructed: dict[Path, GeneralParametrization | JaxComplexArray] = dict(p)
        for info in self.distributions:
            kwargs: dict[str, GeneralParametrization | JaxComplexArray | dict[str, Any]] = {
                    name: constructed[*info.path, name]
                    for name in support(info.type_)}
            sub_distributions = {name: constructed[*info.path, name]
                                 for name in info.sub_distribution_names}
            if sub_distributions:
                kwargs['sub_distributions_objects'] = sub_distributions
            constructed[info.path] = info.type_(**kwargs)
        retval = constructed[()]
        assert isinstance(retval, GeneralParametrization)
        return cast(P, retval)

    def reinterpret(self, q: GeneralParametrization) -> P:
        """Reinterpret one parametrization using the saved structure."""
        p_paths = [(*info.path, name)
                   for info in self.distributions
                   for name in support(info.type_)]
        q_values = parameters(q, support=False).values()
        q_params_as_p = dict(zip(p_paths, q_values, strict=True))
        return self.assemble(q_params_as_p)

    def domain_support(self) -> dict[Path, Support]:
        return {info.path: info.type_.domain_support()
                for info in self.distributions
                if issubclass(info.type_, Parametrization)}

    @classmethod
    def _extract_distributions(cls, p: P) -> list[SubDistributionInfo]:
        return list(cls._walk(cls._make_info, p))

    @classmethod
    def _walk(cls,
              f: Callable[[GeneralParametrization, Path], T],
              q: GeneralParametrization,
              base_path: Path = (),
              ) -> Iterable[T]:
        """Post-order traversal of q."""
        for name, sub_distribution in q.sub_distributions().items():
            this_path = (*base_path, name)
            yield from cls._walk(f, sub_distribution, this_path)
        yield f(q, base_path)

    @classmethod
    def _make_info(cls, q: GeneralParametrization, path: Path) -> SubDistributionInfo:
        from .interfaces.multidimensional import Multidimensional  # noqa: PLC0415
        dimensions = q.dimensions() if isinstance(q, Multidimensional) else 1
        sub_distribution_names = list(q.sub_distributions())
        return SubDistributionInfo(path, type(q), dimensions, sub_distribution_names)


@dataclass
class Flattener(Structure[P]):
    """Flattens a GeneralParametrization into an array of variable parameters.

    Like Structure, it divides the parametrization---in this case, into the flattened parameters and
    the information required to rebuid the parametrization.

    This is useful when implementing machine learning algorithms since the variables parameters are
    the inputs and outputs of neural networks.
    """
    fixed_parameters: dict[Path, JaxComplexArray]
    mapped_to_plane: bool = field(static=True)

    def unflatten(self, flattened: JaxRealArray) -> P:
        """Unflatten an array into a GeneralParametrization.

        Args:
            flattened: The flattened array.
        """
        consumed = 0
        constructed: dict[Path, GeneralParametrization] = {}
        available = flattened.shape[-1]
        for info in self.distributions:
            kwargs: dict[str, GeneralParametrization | JaxComplexArray | dict[str, Any]] = {}
            for name, this_support in support(info.type_, fixed=False).items():
                k = this_support.num_elements(info.dimensions)
                if consumed + k > available:
                    raise ValueError('Incompatible array')  # noqa: TRY003
                kwargs[name] = this_support.unflattened(flattened[..., consumed: consumed + k],
                                                        info.dimensions,
                                                        map_from_plane=self.mapped_to_plane)
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
    def flatten(cls,
                p: P,
                *,
                map_to_plane: bool = True
                ) -> tuple[Self, JaxRealArray]:
        """Flatten a GeneralParametrization.

        Args:
            p: The object to flatten.
            map_to_plane: Whether to map the individual parameters to the plane.  It should be false
                if flattening when the meaning of the flattened parameters should reflect parameter
                values (e.g., when taking a different of expectation parameters).  It should be true
                when passing to a neural network.
        """
        arrays = [x
                  for xs in cls._walk(partial(cls._make_flat, map_to_plane=map_to_plane), p)
                  for x in xs]
        flattened_array = reduce(partial(jnp.append, axis=-1), arrays)
        return (cls(cls._extract_distributions(p),
                    parameters(p, fixed=True),
                    map_to_plane),
                flattened_array)

    @classmethod
    def create_flattener(cls,
                         p: GeneralParametrization,
                         q_cls: type[P],
                         *,
                         mapped_to_plane: bool = True
                         ) -> 'Flattener[P]':
        """Create a Flattener.

        Args:
            p: The object from which to get dimensions and fixed parameters.
            q_cls: The type of the returned Flattener.
            mapped_to_plane: Whether the flattener should map from the plane.
        """
        if p.sub_distributions():
            raise ValueError
        info = cls._make_info(p, path=())
        info = replace(info, type_=q_cls)
        distributions = [info]
        fixed_parameters = parameters(p, fixed=True, support=False)
        return Flattener(distributions, fixed_parameters, mapped_to_plane)

    @classmethod
    def _make_flat(cls, q: GeneralParametrization, path: Path, *, map_to_plane: bool
                   ) -> list[JaxRealArray]:
        return [support.flattened(value, map_to_plane=map_to_plane)
                for value, support in parameters(q, fixed=False, support=True,
                                                 recurse=False).values()]
