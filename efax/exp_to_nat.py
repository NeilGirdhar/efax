from typing import List

import numpy as np
import scipy
from ipromise import implements
from jax import jacfwd
from tjax import RealTensor

from .exponential_family import ExponentialFamily

__all__: List[str] = []


class ExpToNat(ExponentialFamily):
    """
    This mixin implements the conversion from expectation to natural parameters using Newton's
    method with a Jacobian to invert the gradient log-normalizer.
    """

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealTensor) -> RealTensor:
        q = np.empty_like(p)
        for i in np.ndindex(p.shape[: -1]):
            this_p = p[i]

            def f(some_q: RealTensor, this_p: RealTensor = this_p) -> RealTensor:
                some_p = self.nat_to_exp(some_q)
                return some_p - this_p

            solution = scipy.optimize.root(f, np.zeros_like(this_p), tol=1e-5, jac=jacfwd(f))
            if not solution.success:
                raise ValueError("Failed to find natural parmaeters for "
                                 f"{this_p} because {solution.message}.")
            q[i] = solution.x
        return q
