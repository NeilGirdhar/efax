import math
from typing import Optional

import numpy as np
from chex import Array
from ipromise import implements, overrides
from jax import numpy as jnp
from jax.nn import softplus
from tjax import RealArray

from .exp_to_nat import ExpToNat
from .exponential_family import ExponentialFamily

__all__ = ['Logarithmic']


class Logarithmic(ExpToNat, ExponentialFamily):

    def __init__(self) -> None:
        super().__init__(num_parameters=1)

    # Implemented methods --------------------------------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealArray) -> RealArray:
        return jnp.log(-jnp.log1p(-jnp.exp(q[..., 0])))

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealArray) -> RealArray:
        exp_q = jnp.exp(q)
        return jnp.where(q < -100,
                         1.0,
                         exp_q / (jnp.expm1(q) * jnp.log1p(-exp_q)))

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealArray) -> RealArray:
        return x[..., np.newaxis]

    # Overridden methods ---------------------------------------------------------------------------
    @overrides(ExponentialFamily)
    def carrier_measure(self, x: RealArray) -> RealArray:
        return -jnp.log(x)

    @overrides(ExponentialFamily)
    def expected_carrier_measure(self, p: RealArray) -> RealArray:
        raise NotImplementedError

    @overrides(ExpToNat)
    def exp_to_nat_early_out(self, p: Array) -> Optional[Array]:
        if p < 1.0:
            raise ValueError
        if p == 1.0:
            return math.inf
        return None

    @overrides(ExpToNat)
    def exp_to_nat_transform_q(self, transformed_q: Array) -> Array:
        # Run Newton's method on the whole real line.
        return -softplus(transformed_q)
