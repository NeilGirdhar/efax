from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Any, Generic, Self, TypeAlias, TypeVar

from jax import vmap
from tjax import JaxRealArray, jit
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...expectation_parametrization import ExpectationParametrization
from ...natural_parametrization import NaturalParametrization
from ...parametrization import SimpleDistribution
from ...structure import Flattener

NP = TypeVar('NP', bound=NaturalParametrization[Any, Any])
SP: TypeAlias = JaxRealArray


class ExpToNatMinimizer:
    def solve(self, exp_to_nat: ExpToNat[Any]) -> SP:
        raise NotImplementedError


@dataclass
class ExpToNat(ExpectationParametrization[NP], SimpleDistribution, Generic[NP]):
    """This mixin implements the conversion from expectation to natural parameters.

    It uses Newton's method with a Jacobian to invert the gradient log-normalizer.
    """
    _: KW_ONLY
    minimizer: ExpToNatMinimizer | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if hasattr(super(), '__post_init__'):
            super().__post_init__()  # type: ignore # pyright: ignore
        if self.minimizer is None:
            from .optimistix import default_bisection_minimizer, default_minimizer  # noqa: PLC0415
            initial_search_parameters = self.initial_search_parameters()
            object.__setattr__(self,
                               'minimizer',
                               default_minimizer
                               if initial_search_parameters.shape[-1] > 1
                               else default_bisection_minimizer)

    @jit
    @override
    def to_nat(self) -> NP:
        flattener, flattened = Flattener[Self].flatten(self)  # type: ignore[arg-type]

        def solve(flattener: Flattener[Self], flattened: JaxRealArray) -> JaxRealArray:
            x = flattener.unflatten(flattened)
            assert self.minimizer is not None
            return self.minimizer.solve(x)

        for _ in range(self.ndim):
            solve = vmap(solve, in_axes=(None, 0))
        return self.search_to_natural(solve(flattener, flattened))

    def initial_search_parameters(self) -> SP:
        """The initial value of the parameters used by the search algorithm.

        Returns a real array of shape (self.shape, n) for some n.
        """
        _, flattened = Flattener.flatten(self)
        xp = self.array_namespace()
        return xp.zeros_like(flattened)

    def search_to_natural(self, search_parameters: SP) -> NP:
        """Convert the search parameters to the natural parametrization.

        Args:
            search_parameters: The parameters in search space.
        Returns: The corresponding natural parameters.

        This function should be monotonic.
        """
        np_cls = self.natural_parametrization_cls()
        assert issubclass(np_cls, SimpleDistribution)
        flattener = Flattener.create_flattener(self,  # type: ignore[unreachable]
                                               override_unflattened_type=np_cls,
                                               mapped_to_plane=True)
        return flattener.unflatten(search_parameters)

    def search_gradient(self, search_parameters: SP) -> SP:
        """Convert the search parameters to the natural gradient.

        This function converts natural parameters to expectations parameters, takes the difference
        with self, and flattens the result.

        Args:
            search_parameters: The parameters in search space.
        Returns: The gradient of the loss wrt to the search parameters.
        """
        search_np = self.search_to_natural(search_parameters)
        search_ep: ExpToNat[NP] = search_np.to_exp()
        _, self_flat = Flattener.flatten(self, map_to_plane=False)
        _, search_flat = Flattener.flatten(search_ep, map_to_plane=False)
        return search_flat - self_flat
