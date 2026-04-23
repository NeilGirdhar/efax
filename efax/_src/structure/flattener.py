from dataclasses import replace
from functools import partial
from typing import Any, Self, cast, overload

from array_api_compat import array_namespace
from tjax import JaxArray, JaxComplexArray, JaxRealArray
from tjax.dataclasses import dataclass, field
from typing_extensions import TypeVar

from efax._src.iteration import parameters
from efax._src.parametrization import Distribution, SimpleDistribution
from efax._src.types import Path

from .assembler import JointDistributionInfo, SimpleDistributionInfo
from .estimator import Estimator
from .parameter_names import parameter_names
from .parameter_supports import parameter_supports

P = TypeVar("P", bound=Distribution, default=Any)
SP = TypeVar("SP", bound=SimpleDistribution, default=Any)


@dataclass
class Flattener(Estimator[P]):
    """An Estimator that also converts distributions to and from flat arrays.

    Extends Estimator with the ability to encode a Distribution as an array of shape
    (*distribution.shape, k) — where k is final_dimension_size — and decode it back,
    making distributions compatible with neural networks and numerical optimizers.
    Fixed parameters are excluded from the encoded array and reinserted automatically on decode.

    The mapped_to_plane flag controls whether parameters are bijectively mapped from their
    constrained support (e.g., a simplex or positive reals) to all of ℝⁿ.  Set it True when
    interfacing with a neural network (to prevent invalid outputs from causing catastrophic
    failures), and False when the flat values should reflect true parameter magnitudes — for
    example when differencing expectation parameters or computing Fisher information and Jacobians.
    """

    mapped_to_plane: bool = field(static=True)

    def final_dimension_size(self) -> int:
        """Return the size of the last dimension of the flat array produced by flatten."""
        return sum(
            support.num_elements(info.dimensions)
            for info in self.infos
            if isinstance(info, SimpleDistributionInfo)
            for _, support, _ in parameter_supports(info.type_, fixed=False)
        )

    def unflatten(self, flattened: JaxRealArray, *, raveled: bool = False) -> P:
        """Decode a flat array back into a Distribution, reinserting fixed parameters.

        Args:
            flattened: The flat array produced by flatten.
            raveled: If False (default), flattened must have shape
                (..., n_components, params_per_component) as produced by Flattener.flatten.
                If True, flattened may be a 1D vector of shape
                (n_components * params_per_component,) — i.e., the caller has raveled the
                structured flat array.
        """
        xp = array_namespace(flattened)
        if raveled:
            final_dimension_size = self.final_dimension_size()
            if final_dimension_size == 0:
                if flattened.size != 0:
                    raise ValueError("Incompatible array")  # noqa: TRY003
                fixed_parameters = parameters(self.assemble(self.fixed_parameters))
                shape = next(iter(fixed_parameters.values())).shape
                flattened = xp.reshape(flattened, (*shape, 0))
            else:
                flattened = xp.reshape(flattened, (-1, final_dimension_size))
        consumed = 0
        constructed: dict[Path, Distribution] = {}
        available = flattened.shape[-1]
        for info in self.infos:
            if isinstance(info, JointDistributionInfo):
                sub_distributions = {
                    name: constructed[*info.path, name] for name in info.sub_distribution_names
                }
                constructed[info.path] = info.type_(_sub_distributions=sub_distributions)
                continue
            regular_kwargs: dict[str, JaxArray] = {}
            for name, support, value_receptacle in parameter_supports(info.type_, fixed=False):
                k = support.num_elements(info.dimensions)
                if consumed + k > available:
                    raise ValueError("Incompatible array")  # noqa: TRY003
                value = regular_kwargs[name] = support.unflattened(
                    flattened[..., consumed : consumed + k],
                    info.dimensions,
                    map_from_plane=self.mapped_to_plane,
                )
                value_receptacle.set_value(value)
                consumed += k
            kwargs: dict[str, Distribution | JaxComplexArray | dict[str, Any]] = dict(
                regular_kwargs
            )
            for name in parameter_names(info.type_, fixed=True):
                kwargs[name] = self.fixed_parameters[*info.path, name]
            constructed[info.path] = info.type_(**kwargs)
        if consumed != available:
            raise ValueError("Incompatible array")  # noqa: TRY003
        return cast("P", constructed[()])

    @classmethod
    def flatten(cls, p: P, *, mapped_to_plane: bool = True) -> tuple[Self, JaxRealArray]:
        """Encode a Distribution as a (Flattener, flat_array) pair.

        Returns both the flat array and the Flattener needed to decode it.  Fixed parameters
        are stored in the Flattener rather than the array.

        Args:
            p: The distribution to encode.
            mapped_to_plane: Whether to bijectively map constrained parameters to all of ℝⁿ.
                Set False when the flat values should reflect true parameter magnitudes, e.g.,
                when differencing expectation parameters or computing gradients.
                Set True when passing to a neural network.
        """
        xp = array_namespace(p)
        arrays = [
            x
            for xs in cls._walk(partial(cls._make_flat, map_to_plane=mapped_to_plane), p)
            for x in xs
        ]

        flattened_array = xp.concat(arrays, axis=-1) if arrays else xp.empty((*p.shape, 0))
        return (
            cls(cls._extract_distributions(p), parameters(p, fixed=True), mapped_to_plane),
            flattened_array,
        )

    @overload
    @classmethod
    def create_flattener(
        cls, p: SP, *, unflatten_as_type: None = None, mapped_to_plane: bool = True
    ) -> "Flattener[SP]": ...
    @overload
    @classmethod
    def create_flattener(
        cls,
        p: SimpleDistribution,
        *,
        unflatten_as_type: type[SP],
        mapped_to_plane: bool = True,
    ) -> "Flattener[SP]": ...
    @classmethod
    def create_flattener(
        cls,
        p: SimpleDistribution,
        *,
        unflatten_as_type: type[SP] | None = None,
        mapped_to_plane: bool = True,
    ) -> "Flattener[SP]":
        """Create a Flattener from a simple distribution instance.

        Captures p's dimensions and fixed parameters without performing any encoding.
        Use this when you want to reuse the same Flattener across many unflatten calls
        (e.g., inside a training loop) rather than calling flatten each time.

        Args:
            p: The distribution from which to extract dimensions and fixed parameters.
            unflatten_as_type: If provided, the Flattener will decode to this type
                instead of type(p).
            mapped_to_plane: Whether unflatten should apply the inverse plane mapping.
        """
        if p.sub_distributions():
            raise ValueError
        type_ = type(p) if unflatten_as_type is None else unflatten_as_type
        info = cls._make_info(p, path=())
        info = replace(info, type_=type_)
        infos = [info]
        fixed_parameters = parameters(p, fixed=True)
        return Flattener(infos, fixed_parameters, mapped_to_plane)

    @classmethod
    def _make_flat(
        cls, q: Distribution, path: Path, /, *, map_to_plane: bool
    ) -> list[JaxRealArray]:
        retval: list[JaxRealArray] = []
        for value, (_, support, value_receptacle) in zip(
            parameters(q, fixed=False, recurse=False).values(),
            parameter_supports(q, fixed=False),
            strict=True,
        ):
            value_receptacle.set_value(value)
            retval.append(support.flattened(value, map_to_plane=map_to_plane))
        return retval
