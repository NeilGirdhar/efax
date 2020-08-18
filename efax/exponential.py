import numpy as np
from ipromise import implements
from jax import numpy as jnp
from tjax import RealArray

from .exponential_family import ExponentialFamily

__all__ = ['Exponential']


class Exponential(ExponentialFamily):

    def __init__(self) -> None:
        super().__init__(num_parameters=1)

    # Implemented methods --------------------------------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealArray) -> RealArray:
        return -jnp.log(-q[..., 0])

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealArray) -> RealArray:
        return -1.0 / q

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealArray) -> RealArray:
        return -1.0 / p

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealArray) -> RealArray:
        return x[..., np.newaxis]
