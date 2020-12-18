from typing import Any

import numpy as np
from ipromise import implements
from jax import numpy as jnp
from tjax import RealArray

from .exponential_family import ExponentialFamily

__all__ = ['Normal']


class Normal(ExponentialFamily):

    def __init__(self, **kwargs: Any):
        super().__init__(num_parameters=2, **kwargs)

    # Implemented methods --------------------------------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealArray) -> RealArray:
        qa = q[..., 0]
        qb = q[..., 1]
        return -jnp.square(qa) / (4.0 * qb) + 0.5 * jnp.log(-np.pi / qb)

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealArray) -> RealArray:
        qa = q[..., 0]
        qb = q[..., 1]
        ratio = -qa / (2 * qb)
        return jnp.stack([ratio, (jnp.square(ratio) - 0.5 / qb)], axis=-1)

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealArray) -> RealArray:
        mean = p[..., 0]
        second_moment = p[..., 1]
        variance = second_moment - jnp.square(mean)
        return jnp.stack([mean / variance, -0.5 / variance], axis=-1)

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealArray) -> RealArray:
        return jnp.stack([x, jnp.square(x)], axis=-1)
