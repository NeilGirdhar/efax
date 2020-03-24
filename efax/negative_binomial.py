import jax.scipy.special as jss
from ipromise import implements, overrides
from jax import numpy as jnp

from .exponential_family import ExponentialFamily

__all__ = ['NegativeBinomial', 'Geometric']


class NegativeBinomial(ExponentialFamily):

    def __init__(self, r):
        super().__init__(num_parameters=1)
        self.r = r

    # Implemented methods -----------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q):
        return -self.r * jnp.log1p(-jnp.exp(q[..., 0]))

    @implements(ExponentialFamily)
    def nat_to_exp(self, q):
        return self.r / jnp.expm1(-q)

    @implements(ExponentialFamily)
    def exp_to_nat(self, p):
        return -jnp.log1p(self.r / p)

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x):
        return x

    # Overridden methods ------------------------------------------------------
    @overrides(ExponentialFamily)
    def carrier_measure(self, x):
        lgamma = jss.gammaln
        a = x + self.r - 1
        # Return log(a choose x).
        return lgamma(a + 1) - lgamma(x + 1) - lgamma(a - x + 1)

    @overrides(ExponentialFamily)
    def expected_carrier_measure(self, p):
        if self.r == 1:
            return jnp.zeros(self.shape)
        raise NotImplementedError


class Geometric(NegativeBinomial):

    def __init__(self):
        super().__init__(r=1)
