import numpy as np
from ipromise import implements, overrides
from jax import numpy as jnp
from jax.scipy import special as jss
from tjax import RealArray

from .exponential_family import ExponentialFamily

__all__ = ['Poisson']


class Poisson(ExponentialFamily):

    def __init__(self) -> None:
        super().__init__(num_parameters=1)

    # Implemented methods --------------------------------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealArray) -> RealArray:
        # pylint: disable=no-self-use
        return jnp.exp(q)[..., 0]

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealArray) -> RealArray:
        return jnp.exp(q)

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealArray) -> RealArray:
        return jnp.log(p)

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealArray) -> RealArray:
        return x[..., np.newaxis]

    # Overridden methods ---------------------------------------------------------------------------
    @overrides(ExponentialFamily)
    def carrier_measure(self, x: RealArray) -> RealArray:
        return -jss.gammaln(x + 1)

    @overrides(ExponentialFamily)
    def expected_carrier_measure(self, p: RealArray) -> RealArray:
        raise NotImplementedError
