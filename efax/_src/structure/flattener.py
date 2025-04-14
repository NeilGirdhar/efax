from dataclasses import replace
from functools import partial
from typing import Any, Self, TypeVar, cast, overload

from array_api_compat import array_namespace
from tjax import JaxArray, JaxComplexArray, JaxRealArray
from tjax.dataclasses import dataclass, field

from ..iteration import parameters
from ..parametrization import Distribution, SimpleDistribution
from ..types import Path
from .estimator import MaximumLikelihoodEstimator
from .parameter_names import parameter_names
from .parameter_supports import parameter_supports

P = TypeVar('P', bound=Distribution)
SP = TypeVar('SP', bound=SimpleDistribution)


@dataclass
class Flattener(MaximumLikelihoodEstimator[P]):
    """This class can flatten and unflatten distributions.

    The flattener can optionally map the values to and from the full plane.  This is useful when
    implementing machine learning algorithms since the variables parameters are the inputs and
    outputs of neural networks.  We don't want catastrophic results when the neural network produces
    values that are outside the support.

    Examples where we want to disable mapping to the plane include:
    * The ExpToNat subtracts flattened expectation parameters to calculate the gradient wrt to
      the natural parameters.
    * The Fisher information matrix and the Jacobian matrix are calculated wrt to flattened
      parameters, and flattened outputs.
    """
    mapped_to_plane: bool = field(static=True)

    def final_dimension_size(self) -> int:
        return sum(support.num_elements(info.dimensions)
                   for info in self.infos
                   for _, support, _ in parameter_supports(info.type_, fixed=False))

    def unflatten(self, flattened: JaxRealArray, *, return_vector: bool = False) -> P:
        """Unflatten an array into a Distribution.

        Args:
            flattened: The flattened array.
            return_vector: If true, reshape the array so that a vector is returned.
        """
        xp = array_namespace(flattened)
        if return_vector:
            flattened = xp.reshape(flattened, (-1, self.final_dimension_size()))
        consumed = 0
        constructed: dict[Path, Distribution] = {}
        available = flattened.shape[-1]
        for info in self.infos:
            regular_kwargs: dict[str, JaxArray] = {}
            for name, support, value_receptacle in parameter_supports(info.type_, fixed=False):
                k = support.num_elements(info.dimensions)
                if consumed + k > available:
                    raise ValueError('Incompatible array')  # noqa: TRY003
                value = regular_kwargs[name] = support.unflattened(
                        flattened[..., consumed: consumed + k],
                        info.dimensions,
                        map_from_plane=self.mapped_to_plane)
                value_receptacle.set_value(value)
                consumed += k
            kwargs: dict[str, Distribution | JaxComplexArray | dict[str, Any]] = dict(
                    regular_kwargs)
            for name in parameter_names(info.type_, fixed=True):
                kwargs[name] = self.fixed_parameters[*info.path, name]
            sub_distributions = {name: constructed[*info.path, name]
                                 for name in info.sub_distribution_names}
            if sub_distributions:
                kwargs['_sub_distributions'] = sub_distributions
            constructed[info.path] = info.type_(**kwargs)
        if consumed != available:
            raise ValueError('Incompatible array')  # noqa: TRY003
        return cast('P', constructed[()])

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
        xp = p.array_namespace()
        arrays = [x
                  for xs in cls._walk(partial(cls._make_flat, map_to_plane=map_to_plane), p)
                  for x in xs]

        flattened_array = xp.concat(arrays, axis=-1)
        return (cls(cls._extract_distributions(p),
                    parameters(p, fixed=True),
                    map_to_plane),
                flattened_array)

    @overload
    @classmethod
    def create_flattener(cls,
                         p: SP,
                         *,
                         override_unflattened_type: None = None,
                         mapped_to_plane: bool = True
                         ) -> 'Flattener[SP]': ...
    @overload
    @classmethod
    def create_flattener(cls,
                         p: SimpleDistribution,
                         *,
                         override_unflattened_type: type[SP],
                         mapped_to_plane: bool = True
                         ) -> 'Flattener[SP]': ...
    @classmethod
    def create_flattener(cls,
                         p: SimpleDistribution,
                         *,
                         override_unflattened_type: type[SP] | None = None,
                         mapped_to_plane: bool = True
                         ) -> 'Flattener[SP]':
        """Create a Flattener.

        Args:
            p: The object from which to get dimensions and fixed parameters.
            override_unflattened_type: The type of the returned Flattener.
            mapped_to_plane: Whether the flattener should map from the plane.
        """
        if p.sub_distributions():
            raise ValueError
        type_ = type(p) if override_unflattened_type is None else override_unflattened_type
        info = cls._make_info(p, path=())
        info = replace(info, type_=type_)
        infos = [info]
        fixed_parameters = parameters(p, fixed=True)
        return Flattener(infos, fixed_parameters, mapped_to_plane)

    @classmethod
    def _make_flat(cls, q: Distribution, path: Path, /, *, map_to_plane: bool
                   ) -> list[JaxRealArray]:
        retval: list[JaxRealArray] = []
        for value, (_, support, value_receptacle) in zip(
                parameters(q, fixed=False, recurse=False).values(),
                parameter_supports(q, fixed=False),
                strict=True):
            value_receptacle.set_value(value)
            retval.append(support.flattened(value, map_to_plane=map_to_plane))
        return retval
