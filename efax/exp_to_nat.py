from typing import List, Optional

import numpy as np
import scipy
from chex import Array
from ipromise import implements
from jax import jacfwd

from .exponential_family import ExponentialFamily

__all__: List[str] = []


class ExpToNat(ExponentialFamily):
    """
    This mixin implements the conversion from expectation to natural parameters using Newton's
    method with a Jacobian to invert the gradient log-normalizer.
    """

    # Implemented methods --------------------------------------------------------------------------
    @implements(ExponentialFamily)
    def exp_to_nat(self, p: Array) -> Array:
        q = np.empty_like(p)
        for i in np.ndindex(p.shape[: -1]):
            this_p = p[i]
            this_q = self.exp_to_nat_early_out(this_p)
            if this_q is None:
                def f(transformed_q: Array, this_p: Array = this_p) -> Array:
                    some_p = self.nat_to_exp(self.exp_to_nat_transform_q(transformed_q))
                    print(f"Trying {self.exp_to_nat_transform_q(transformed_q)}, which gives "
                          f"{some_p} -> {this_p}")
                    return some_p - this_p

                solution = scipy.optimize.root(
                    f, self.exp_to_nat_initial_q(this_p), tol=1e-5, jac=jacfwd(f))
                if not solution.success:
                    raise ValueError("Failed to find natural parmaeters for "
                                     f"{this_p} because {solution.message}.")
                this_q = self.exp_to_nat_transform_q(solution.x)
            q[i] = this_q
        return q

    # New methods ----------------------------------------------------------------------------------
    def exp_to_nat_early_out(self, p: Array) -> Optional[Array]:
        """
        Args:
            p: The expectation parameters passed to exp_to_nat having shape (self.num_parameters,).
        Returns: q if there's an easy answer to exp_to_nat, otherwise None.
        """
        return None

    def exp_to_nat_initial_q(self, p: Array) -> Array:
        """
        Args:
            p: The expectation parameters passed to exp_to_nat having shape (self.num_parameters,).
        Returns: transformed_q, a form of the natural parameters that Newton's method will run on.
        """
        return np.zeros_like(p)

    def exp_to_nat_transform_q(self, transformed_q: Array) -> Array:
        """
        Args:
            transformed_q: A form of the natural parameters that Newton's method will run on having
                shape (self.num_parameters,).
        Returns: q, the natural parameters.
        """
        return transformed_q
