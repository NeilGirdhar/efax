import math

import jax.numpy as jnp
import numpy as np
import scipy.optimize
from ipromise import implements, overrides

from .exponential_family import ExponentialFamily
from .tensors import RealTensor

__all__ = ['Logarithmic']


class Logarithmic(ExponentialFamily):

    def __init__(self):
        super().__init__(num_parameters=1)

    # Implemented methods -----------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealTensor) -> RealTensor:
        return jnp.log(-jnp.log1p(-jnp.exp(q[..., 0])))

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealTensor) -> RealTensor:
        exp_q = jnp.exp(q)
        return jnp.where(
            q < -100,
            1.0,
            exp_q / (jnp.expm1(q) * jnp.log1p(-exp_q)))

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealTensor) -> RealTensor:
        q = np.empty_like(p)
        it = np.nditer([p, q],
                       op_flags=[['readonly'], ['writeonly', 'allocate']])
        with it:
            for this_p, this_q in it:
                # Work on the transformed problem.  Find log(-q), which is
                # over the whole real line.
                if this_p < 1.0:
                    raise ValueError
                if this_p == 1.0:
                    return math.inf

                def f(log_minus_q, this_p=this_p):
                    return (self.nat_to_exp(-math.exp(log_minus_q))
                            - this_p)

                solution = scipy.optimize.root(f,
                                               0.0,
                                               tol=1e-5)
                if not solution.success:
                    raise ValueError(
                        "Failed to find natural parmaeters for "
                        f"{this_p} because {solution.message}.")
                this_q[...] = -math.exp(solution.x)
        return q

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealTensor) -> RealTensor:
        return x

    # Overridden methods ------------------------------------------------------
    @overrides(ExponentialFamily)
    def carrier_measure(self, x: RealTensor) -> RealTensor:
        return -jnp.log(x)

    @overrides(ExponentialFamily)
    def expected_carrier_measure(self, p: RealTensor) -> RealTensor:
        raise NotImplementedError
