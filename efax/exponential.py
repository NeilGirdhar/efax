from ipromise import implements
from jax import numpy as jnp

from .exponential_family import ExponentialFamily

__all__ = ['Exponential']


class Exponential(ExponentialFamily):

    def __init__(self):
        super().__init__(num_parameters=1)

    # Implemented methods -----------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q):
        return -jnp.log(-q[..., 0])

    @implements(ExponentialFamily)
    def nat_to_exp(self, q):
        return -1.0 / q

    @implements(ExponentialFamily)
    def exp_to_nat(self, p):
        return -1.0 / p

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x):
        return x
