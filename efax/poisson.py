import jax.numpy as jnp
import jax.scipy.special as jss
import numpy as np
from ipromise import implements, overrides

from .exponential_family import ExponentialFamily
from .tensors import RealTensor

__all__ = ['Poisson']


class Poisson(ExponentialFamily):

    def __init__(self):
        super().__init__(num_parameters=1)

    # Implemented methods -----------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealTensor) -> RealTensor:
        # pylint: disable=no-self-use
        return jnp.exp(q)[..., 0]

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealTensor) -> RealTensor:
        return jnp.exp(q)

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealTensor) -> RealTensor:
        return jnp.log(p)

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealTensor) -> RealTensor:
        return x[..., np.newaxis]

    # Overridden methods ------------------------------------------------------
    @overrides(ExponentialFamily)
    def carrier_measure(self, x: RealTensor) -> RealTensor:
        return -jss.gammaln(x + 1)

    @overrides(ExponentialFamily)
    def expected_carrier_measure(self, p: RealTensor) -> RealTensor:
        raise NotImplementedError
