from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, cast

import jax.numpy as jnp
from numpy.random import Generator
from tjax import JaxArray, JaxComplexArray, JaxRealArray, Shape
from tjax.dataclasses import dataclass, field

from .iteration import flatten_mapping, parameters, support
from .parameter import Support
from .parametrization import Distribution, SimpleDistribution
from .types import Path

if TYPE_CHECKING:
    from .natural_parametrization import NaturalParametrization


@dataclass
class SubDistributionInfo:
    """A hashable collection of the static information for recreating sub-distributions."""
    path: Path
    type_: type[Distribution]
    dimensions: int
    sub_distribution_names: list[str]


T = TypeVar('T')
P = TypeVar('P', bound=Distribution)


@dataclass
class Structure(Generic[P]):
    """Structure generalizes the notion of type for Distribution objects.

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
        from .expectation_parametrization import ExpectationParametrization  # noqa: PLC0415
        infos = []
        for info in self.infos:
            assert issubclass(info.type_, ExpectationParametrization)
            infos.append(SubDistributionInfo(info.path, info.type_.natural_parametrization_cls(),
                                             info.dimensions, info.sub_distribution_names))
        return Structure(infos)

    def to_exp(self) -> 'Structure[Any]':
        from .natural_parametrization import NaturalParametrization  # noqa: PLC0415
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
                    for name in support(info.type_)}
            sub_distributions = {name: constructed[*info.path, name]
                                 for name in info.sub_distribution_names}
            if sub_distributions:
                kwargs['_sub_distributions'] = sub_distributions
            constructed[info.path] = info.type_(**kwargs)
        retval = constructed[()]
        assert isinstance(retval, Distribution)
        return cast(P, retval)

    def reinterpret(self, q: Distribution) -> P:
        """Reinterpret a distribution in one parametrization using the saved structure."""
        p_paths = [(*info.path, name)
                   for info in self.infos
                   for name in support(info.type_)]
        q_values = parameters(q, support=False).values()
        q_params_as_p = dict(zip(p_paths, q_values, strict=True))
        return self.assemble(q_params_as_p)

    def domain_support(self) -> dict[Path, Support]:
        """Get the domain support, which may require the structure for joint distributions."""
        return {info.path: info.type_.domain_support()
                for info in self.infos
                if issubclass(info.type_, SimpleDistribution)}

    def generate_random(self, rng: Generator, shape: Shape) -> P:
        """Generate a random distribution."""
        path_and_values = {}
        for info in self.infos:
            kwargs: dict[str, JaxComplexArray] = {}
            for name in support(info.type_):
                s = info.type_.adjust_support(name, **kwargs)
                value = s.generate(rng, shape, info.dimensions)
                path_and_values[(*info.path, name)] = kwargs[name] = value
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
        from .interfaces.multidimensional import Multidimensional  # noqa: PLC0415
        dimensions = q.dimensions() if isinstance(q, Multidimensional) else 1
        sub_distribution_names = list(q.sub_distributions())
        return SubDistributionInfo(path, type(q), dimensions, sub_distribution_names)


@dataclass
class MaximumLikelihoodEstimator(Structure[P]):
    """MaximumLikelihoodEstimator does maximum likelihood estimation.

    To do this, it needs to store the structure and the fixed parameters.
    """
    fixed_parameters: dict[Path, JaxComplexArray]

    @classmethod
    def create_simple_estimator(cls,
                                type_p: type[SimpleDistribution],
                                **fixed_parameters: Any
                                ) -> 'MaximumLikelihoodEstimator[Any]':
        return MaximumLikelihoodEstimator(
                [SubDistributionInfo((), type_p, 0, [])],
                {(name,): value for name, value in fixed_parameters.items()})

    @classmethod
    def create_estimator(cls, p: P) -> Self:
        infos = cls.create(p).infos
        fixed_parameters = parameters(p, fixed=True)
        return cls(infos, fixed_parameters)

    @classmethod
    def create_estimator_from_natural(cls, p: 'NaturalParametrization[Any, Any]'
                                      ) -> 'MaximumLikelihoodEstimator[Any]':
        infos = MaximumLikelihoodEstimator.create(p).to_exp().infos
        fixed_parameters = parameters(p, fixed=True)
        return cls(infos, fixed_parameters)

    def sufficient_statistics(self, x: dict[str, Any] | JaxComplexArray) -> P:
        from .expectation_parametrization import ExpectationParametrization  # noqa: PLC0415
        from .natural_parametrization import NaturalParametrization  # noqa: PLC0415
        from .transform.joint import JointDistributionE  # noqa: PLC0415
        constructed: dict[Path, ExpectationParametrization[Any]] = {}

        def g(info: SubDistributionInfo, x: JaxComplexArray) -> None:
            assert not info.sub_distribution_names
            exp_cls = info.type_
            assert issubclass(exp_cls, ExpectationParametrization)

            nat_cls = exp_cls.natural_parametrization_cls()
            assert issubclass(nat_cls, NaturalParametrization)

            fixed_parameters: dict[str, Any] = {name: self.fixed_parameters[(*info.path, name)]
                                                for name in support(nat_cls, fixed=True)}
            p = nat_cls.sufficient_statistics(x, **fixed_parameters)
            assert isinstance(p, ExpectationParametrization)
            constructed[info.path] = p

        def h(info: SubDistributionInfo) -> None:
            exp_cls = info.type_
            assert issubclass(exp_cls, JointDistributionE)
            sub_distributions = {name: constructed[*info.path, name]
                                 for name in info.sub_distribution_names}
            constructed[info.path] = exp_cls(sub_distributions)

        if isinstance(x, dict):
            flat_x = flatten_mapping(x)
            for info in self.infos:
                if info.path in flat_x:
                    g(info, flat_x[info.path])
                else:
                    h(info)
        else:
            info, = self.infos
            g(info, x)
        return cast(P, constructed[()])


@dataclass
class Flattener(MaximumLikelihoodEstimator[P]):
    """Flattener can flatten and unflatten distributions.

    The flattener has two fundamental modes:
    * Map-to-plane: This is useful when implementing machine learning algorithms since the variables
      parameters are the inputs and outputs of neural networks.  We don't want catastrophic results
      when the neural network produces values that are outside the support.
    * Examples where map-to-plane is not used include:
        * The ExpToNat subtracts flattened expectation parameters to calculate the gradient wrt to
          the natural parameters.
        * The Fisher information matrix and the Jacobian matrix are calculated wrt to flattened
          parameters, and flattened outputs.
    """
    mapped_to_plane: bool = field(static=True)

    def unflatten(self, flattened: JaxRealArray) -> P:
        """Unflatten an array into a Distribution.

        Args:
            flattened: The flattened array.
        """
        consumed = 0
        constructed: dict[Path, Distribution] = {}
        available = flattened.shape[-1]
        for info in self.infos:
            regular_kwargs: dict[str, JaxArray] = {}
            for name in support(info.type_, fixed=False):
                this_support = info.type_.adjust_support(name, **regular_kwargs)
                k = this_support.num_elements(info.dimensions)
                if consumed + k > available:
                    raise ValueError('Incompatible array')  # noqa: TRY003
                regular_kwargs[name] = this_support.unflattened(
                        flattened[..., consumed: consumed + k],
                        info.dimensions,
                        map_from_plane=self.mapped_to_plane)
                consumed += k
            kwargs: dict[str, Distribution | JaxComplexArray | dict[str, Any]] = dict(
                    regular_kwargs)
            for name in support(info.type_, fixed=True):
                kwargs[name] = self.fixed_parameters[*info.path, name]
            sub_distributions = {name: constructed[*info.path, name]
                                 for name in info.sub_distribution_names}
            if sub_distributions:
                kwargs['_sub_distributions'] = sub_distributions
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
        """Flatten a Distribution.

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
                         p: SimpleDistribution,
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
        infos = [info]
        fixed_parameters = parameters(p, fixed=True, support=False)
        return Flattener(infos, fixed_parameters, mapped_to_plane)

    @classmethod
    def _make_flat(cls, q: Distribution, path: Path, /, *, map_to_plane: bool
                   ) -> list[JaxRealArray]:
        kwargs: dict[str, JaxComplexArray] = {}
        retval: list[JaxRealArray] = []
        for name, value in parameters(q, fixed=False, support=False, recurse=False).items():
            support_ = q.adjust_support(name, **kwargs)
            kwargs[name] = value
            retval.append(support_.flattened(value, map_to_plane=map_to_plane))
        return retval
