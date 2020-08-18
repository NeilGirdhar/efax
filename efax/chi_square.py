import numpy as np
from ipromise import implements, overrides
from jax import numpy as jnp
from jax.scipy import special as jss
from tjax import RealArray

from .exp_to_nat import ExpToNat
from .exponential_family import ExponentialFamily

__all__ = ['ChiSquare']


class ChiSquare(ExpToNat, ExponentialFamily):
    """
    The chi-square distribution with k degrees of freedom is the gamma distribution with shape k/2
    and rate 1/2.

    It has natural parameter k/2 - 1.
    """

    def __init__(self) -> None:
        super().__init__(num_parameters=1)

    # Implemented methods --------------------------------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealArray) -> RealArray:
        k_over_two = q[..., 0] + 1.0
        return jss.gammaln(k_over_two) - k_over_two * jnp.log(0.5)

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealArray) -> RealArray:
        k_over_two = q + 1.0
        return jss.digamma(k_over_two) - jnp.log(0.5)

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealArray) -> RealArray:
        return jnp.log(x)[..., np.newaxis]

    @overrides(ExponentialFamily)
    def carrier_measure(self, x: RealArray) -> RealArray:
        return -x * 0.5

    @overrides(ExponentialFamily)
    def expected_carrier_measure(self, p: RealArray) -> RealArray:
        # Should return k/2, which requires the inverse of the digamma function.
        raise NotImplementedError
