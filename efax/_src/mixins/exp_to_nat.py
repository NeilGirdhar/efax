from __future__ import annotations

from typing import Any, Generic, TypeAlias, TypeVar

import jax.numpy as jnp
from jax import jit
# from tensorflow_probability.substrates.jax.optimizer import lbfgs_minimize
from jaxopt import LBFGS
from tjax import JaxRealArray
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import NaturalParametrization
from ..structure import Flattener

NP = TypeVar('NP', bound=NaturalParametrization[Any, Any])
SP: TypeAlias = JaxRealArray


class ExpToNat(ExpectationParametrization[NP], Generic[NP]):
    """This mixin implements the conversion from expectation to natural parameters.

    It uses LBFGS with a Jacobian to invert the gradient log-normalizer.
    """
    @override
    def to_nat(self) -> NP:
        @jit
        def f(x: SP, /) -> tuple[JaxRealArray, JaxRealArray]:
            g = self.search_gradient(x)
            return 0.5 * jnp.sum(jnp.square(g)), g

        solver = LBFGS(f, value_and_grad=True, tol=1e-5, maxiter=100)
        results = solver.run(self.initial_search_parameters())
        return self.search_to_natural(results.params)

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
