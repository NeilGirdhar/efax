import numpy as np
import scipy
from ipromise import implements, overrides
from jax import numpy as jnp
from jax.scipy import special as jss
from scipy.special import polygamma
from tjax import RealTensor

from .exponential_family import ExponentialFamily

__all__ = ['ChiSquare']


def trigamma(x: RealTensor) -> RealTensor:
    return polygamma(1, x)


class ChiSquare(ExponentialFamily):
    """
    The chi-square distribution with k degrees of freedom is the gamma distribution with shape k/2
    and rate 1/2.

    It has natural parameter k/2 - 1.
    """

    def __init__(self) -> None:
        super().__init__(num_parameters=1)

    # Implemented methods --------------------------------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealTensor) -> RealTensor:
        k_over_two = q[..., 0] + 1.0
        return jss.gammaln(k_over_two) - k_over_two * jnp.log(0.5)

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealTensor) -> RealTensor:
        k_over_two = q + 1.0
        return jss.digamma(k_over_two) - jnp.log(0.5)

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealTensor) -> RealTensor:
        q = np.empty_like(p)
        for i in np.ndindex(p.shape[: -1]):
            this_p = p[i]

            def f(some_q: RealTensor, this_p: RealTensor = this_p) -> RealTensor:
                some_p = self.nat_to_exp(some_q)
                return some_p - this_p

            solution = scipy.optimize.root(f, np.zeros_like(this_p), tol=1e-5)
            if not solution.success:
                raise ValueError("Failed to find natural parmaeters for "
                                 f"{this_p} because {solution.message}.")
            q[i] = solution.x
        return q

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealTensor) -> RealTensor:
        return jnp.log(x)[..., np.newaxis]

    @overrides(ExponentialFamily)
    def carrier_measure(self, x: RealTensor) -> RealTensor:
        return -x * 0.5

    @overrides(ExponentialFamily)
    def expected_carrier_measure(self, p: RealTensor) -> RealTensor:
        # Should return k/2, which requires the inverse of the digamma function.
        raise NotImplementedError
