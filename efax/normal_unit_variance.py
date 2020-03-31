import math

import jax.numpy as jnp
from ipromise import implements, overrides

from .exponential_family import ExponentialFamily
from .tensors import RealTensor, real_dtype

__all__ = ['NormalUnitVariance']


class NormalUnitVariance(ExponentialFamily):

    # Magic methods -----------------------------------------------------------
    def __repr__(self):
        return (f"{type(self).__name__}(shape={self.shape}, "
                f"num_parameters={self.num_parameters})")

    # Implemented methods -----------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealTensor) -> RealTensor:
        return 0.5 * (jnp.sum(jnp.square(q), axis=-1)
                      + self.num_parameters * math.log(math.pi * 2.0))

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealTensor) -> RealTensor:
        return q

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealTensor) -> RealTensor:
        return p

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealTensor) -> RealTensor:
        return x

    # Overridden methods ------------------------------------------------------
    @overrides(ExponentialFamily)
    def scaled_cross_entropy(self,
                             k: real_dtype,
                             kp: RealTensor,
                             q: RealTensor) -> RealTensor:
        p = kp / k
        return jnp.where(
            k == 0.0,
            0.0,
            0.5 * (jnp.sum((kp - k * q) * (p - q), axis=-1)
                   + k * (jnp.log(math.pi * 2.0) + 1.0)))

    @overrides(ExponentialFamily)
    def carrier_measure(self, x: RealTensor) -> RealTensor:
        # The second moment of a delta distribution at x.
        return -0.5 * jnp.square(x)

    @overrides(ExponentialFamily)
    def expected_carrier_measure(self, p: RealTensor) -> RealTensor:
        # The second moment of a normal distribution with mean p.
        return -0.5 * (jnp.sum(jnp.square(p), axis=-1) + 1.0)
