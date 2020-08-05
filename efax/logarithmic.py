import math
from jax.nn import softplus
from typing import Optional

import numpy as np
from ipromise import implements, overrides
from jax import numpy as jnp
from tjax import RealTensor, Tensor

from .exp_to_nat import ExpToNat
from .exponential_family import ExponentialFamily

__all__ = ['Logarithmic']


class Logarithmic(ExpToNat, ExponentialFamily):

    def __init__(self) -> None:
        super().__init__(num_parameters=1)

    # Implemented methods --------------------------------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealTensor) -> RealTensor:
        return jnp.log(-jnp.log1p(-jnp.exp(q[..., 0])))

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealTensor) -> RealTensor:
        exp_q = jnp.exp(q)
        return jnp.where(q < -100,
                         1.0,
                         exp_q / (jnp.expm1(q) * jnp.log1p(-exp_q)))

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealTensor) -> RealTensor:
        return x[..., np.newaxis]

    # Overridden methods ---------------------------------------------------------------------------
    @overrides(ExponentialFamily)
    def carrier_measure(self, x: RealTensor) -> RealTensor:
        return -jnp.log(x)

    @overrides(ExponentialFamily)
    def expected_carrier_measure(self, p: RealTensor) -> RealTensor:
        raise NotImplementedError

    @overrides(ExpToNat)
    def exp_to_nat_early_out(self, p: Tensor) -> Optional[Tensor]:
        if p < 1.0:
            raise ValueError
        if p == 1.0:
            return math.inf
        return None

    @overrides(ExpToNat)
    def exp_to_nat_transform_q(self, transformed_q: Tensor) -> Tensor:
        # Run Newton's method on the whole real line.
        return -softplus(transformed_q)
