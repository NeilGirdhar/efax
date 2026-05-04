from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Any, Generic, Self, override

from array_api_compat import array_namespace
from jax import vmap
from tjax import JaxRealArray, jit
from tjax.dataclasses import dataclass
from typing_extensions import TypeVar

from efax._src.expectation_parametrization import ExpectationParametrization
from efax._src.natural_parametrization import NaturalParametrization
from efax._src.parametrization import SimpleDistribution
from efax._src.structure.flattener import Flattener

NP = TypeVar("NP", bound=NaturalParametrization, default=Any)
type SP = JaxRealArray


class ExpToNatMinimizer:
    """Strategy that solves the EP-to-NP root-finding problem for an ExpToNat distribution."""

    def solve(self, exp_to_nat: ExpToNat) -> SP:
        """Solve for the natural parameters corresponding to exp_to_nat's expectation parameters.

        Returns a flat array of search parameters (natural parameters in flattened form).
        """
        raise NotImplementedError


@dataclass
class ExpToNat(ExpectationParametrization[NP], SimpleDistribution, Generic[NP]):
    """This mixin implements the conversion from expectation to natural parameters.

    It uses Newton's method with a Jacobian to invert the gradient log-normalizer.
    """

    _: KW_ONLY
    minimizer: ExpToNatMinimizer | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Select a default minimizer based on the dimensionality of the search space.
        # Bisection is used for scalar problems; Newton's method for vector problems.
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # ty: ignore
        if self.minimizer is None:
            from .optimistix import default_bisection_minimizer, default_minimizer  # noqa: PLC0415

            initial_search_parameters = self.initial_search_parameters()
            object.__setattr__(
                self,
                "minimizer",
                default_minimizer
                if initial_search_parameters.shape[-1] > 1
                else default_bisection_minimizer,
            )

    @jit
    @override
    def to_nat(self) -> NP:
        """Return the natural parameters by numerically inverting the gradient log-normalizer."""
        flattener, flattened = Flattener[Self].flatten(self)

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
        xp = array_namespace(self)
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
        flattener = Flattener.create_flattener(
            self,
            unflatten_as_type=np_cls,
            mapped_to_plane=True,
        )
        return flattener.unflatten(search_parameters)  # ty: ignore

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
        _, self_flat = Flattener.flatten(self, mapped_to_plane=False)
        _, search_flat = Flattener.flatten(search_ep, mapped_to_plane=False)
        return search_flat - self_flat
