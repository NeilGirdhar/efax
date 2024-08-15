from __future__ import annotations

from dataclasses import KW_ONLY
from typing import Any, Generic, TypeAlias, TypeVar

import jax.numpy as jnp
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
    minimizer: ExpToNatMinimizer | None = None

    def __post_init__(self) -> None:
        if hasattr(super(), '__post_init__'):
            super().__post_init__()  # pyright: ignore
        if self.minimizer is None:
            from .iterated_function import default_minimizer  # noqa: PLC0415
            object.__setattr__(self, 'minimizer', default_minimizer)

    @jit
    @override
    def to_nat(self) -> NP:
        assert self.minimizer is not None
        return self.search_to_natural(self.minimizer.solve(self))

    def initial_search_parameters(self) -> SP:
        """The initial value of the parameters used by the search algorithm.

        Returns a real array of shape (self.shape, n) for some n.
        """
        _, flattened = Flattener.flatten(self)
        return jnp.zeros_like(flattened)

    def search_to_natural(self, search_parameters: SP) -> NP:
        """Convert the search parameters to the natural parametrization.

        Args:
            search_parameters: The parameters in search space.
        Returns: The corresponding natural parameters.

        This function should be monotonic.
        """
        np_cls = self.natural_parametrization_cls()
        flattener = Flattener.create_flattener(self, np_cls, mapped_to_plane=True)
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
