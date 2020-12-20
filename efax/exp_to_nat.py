from functools import partial
from typing import Any, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import scipy
from chex import Array
from jax import jacfwd, jit
from jax.tree_util import tree_map, tree_reduce

from .exponential_family import ExpectationParametrization, NaturalParametrization

__all__: List[str] = []


NP = TypeVar('NP', bound=NaturalParametrization[Any])


class ExpToNat(ExpectationParametrization[NP], Generic[NP]):
    """
    This mixin implements the conversion from expectation to natural parameters using Newton's
    method with a Jacobian to invert the gradient log-normalizer.
    """

    # Implemented methods --------------------------------------------------------------------------
    def to_nat(self) -> NP:
        q = self.initial_flattened_natural()
        i: Tuple[int, ...]
        for i in np.ndindex(self.shape()):
            this_p = tree_map(lambda x, i=i: x[i], self)  # type: ignore
            this_q = this_p.to_nat_early_out()
            if this_q is None:
                flattened_expectation_p = this_p.flatten_expectation()
                flattened_natural_q = this_p.initial_flattened_natural()

                bound_f = partial(type(self)._to_nat_helper, flattened_expectation_p)
                bound_jf = partial(type(self)._to_nat_jacobian, flattened_expectation_p)

                solution = scipy.optimize.root(bound_f, flattened_natural_q, tol=1e-5,
                                               jac=bound_jf)
                if not solution.success:
                    raise ValueError("Failed to find natural parmaeters for "
                                     f"{this_p} because {solution.message}.")
                this_q = solution.x
            q[i] = this_q
        return self.unflatten_natural(q)

    # Non-final methods ----------------------------------------------------------------------------
    def to_nat_early_out(self) -> Optional[Array]:
        """
        Args:
            p: The expectation parameters passed to exp_to_nat.
        Returns: The corresponding flattened natural parameters if there's an easy answer to to_nat,
            otherwise None.
        """
        return None

    def initial_flattened_natural(self) -> Array:
        """
        Returns: A form of the natural parameters that Newton's method will run on.
        """
        return tree_reduce(np.concatenate, tree_map(np.zeros_like, self))

    @classmethod
    def unflatten_natural(cls, flattened_natural: Array) -> NP:
        """
        Args:
            flattened_natural: A form of the natural parameters that Newton's method will run on.
        Returns: The corresponding natural parametrization.
        """
        raise NotImplementedError

    def flatten_expectation(self) -> Array:
        """
        Returns: A form of the expectation parameters that Newton's method will run on.
        """
        raise NotImplementedError

    # Private methods ------------------------------------------------------------------------------
    @classmethod
    @partial(jit, static_argnums=(0,))
    def _to_nat_helper(cls, flattened_expectation_p: Array, flattened_natural_q: Array) -> Array:
        natural_q = cls.unflatten_natural(flattened_natural_q)
        expectation_q = natural_q.to_exp()
        flattened_expectation_q = expectation_q.flatten_expectation()
        return flattened_expectation_q - flattened_expectation_p

    @classmethod
    @partial(jit, static_argnums=(0,))
    def _to_nat_jacobian(cls, flattened_expectation_p: Array, flattened_natural_q: Array) -> Array:
        return jacfwd(cls._to_nat_helper, argnums=1)(flattened_expectation_p, flattened_natural_q)
