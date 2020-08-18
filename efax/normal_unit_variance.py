import math

import numpy as np
from ipromise import implements, overrides
from jax import numpy as jnp
from tjax import RealArray, real_dtype

from .exponential_family import ExponentialFamily

__all__ = ['NormalUnitVariance']


class NormalUnitVariance(ExponentialFamily):

    # Magic methods --------------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"{type(self).__name__}(shape={self.shape}, "
                f"num_parameters={self.num_parameters})")

    # Implemented methods --------------------------------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealArray) -> RealArray:
        return 0.5 * (jnp.sum(jnp.square(q), axis=-1)
                      + self.num_parameters * math.log(math.pi * 2.0))

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealArray) -> RealArray:
        return q

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealArray) -> RealArray:
        return p

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealArray) -> RealArray:
        return x[..., np.newaxis]

    # Overridden methods ---------------------------------------------------------------------------
    @overrides(ExponentialFamily)
    def scaled_cross_entropy(self,
                             k: real_dtype,
                             kp: RealArray,
                             q: RealArray) -> RealArray:
        p = kp / k
        return jnp.where(k == 0.0,
                         0.0,
                         0.5 * (jnp.sum((kp - k * q) * (p - q), axis=-1)
                                + k * (jnp.log(math.pi * 2.0) + 1.0)))

    @overrides(ExponentialFamily)
    def carrier_measure(self, x: RealArray) -> RealArray:
        # The second moment of a delta distribution at x.
        return -0.5 * jnp.square(x)

    @overrides(ExponentialFamily)
    def expected_carrier_measure(self, p: RealArray) -> RealArray:
        # The second moment of a normal distribution with mean p.
        return -0.5 * (jnp.sum(jnp.square(p), axis=-1) + 1.0)
