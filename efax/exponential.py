import jax.numpy as jnp
from ipromise import implements

from .exponential_family import ExponentialFamily
from .tensors import RealTensor

__all__ = ['Exponential']


class Exponential(ExponentialFamily):

    def __init__(self):
        super().__init__(num_parameters=1)

    # Implemented methods -----------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealTensor) -> RealTensor:
        return -jnp.log(-q[..., 0])

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealTensor) -> RealTensor:
        return -1.0 / q

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealTensor) -> RealTensor:
        return -1.0 / p

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealTensor) -> RealTensor:
        return x
