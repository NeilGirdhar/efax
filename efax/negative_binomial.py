from typing import Any

import numpy as np
from ipromise import implements, overrides
from jax import numpy as jnp
from jax.scipy import special as jss
from tjax import RealArray

from .exponential_family import ExponentialFamily

__all__ = ['NegativeBinomial', 'Geometric']


class NegativeBinomial(ExponentialFamily):

    def __init__(self, r: int):
        super().__init__(num_parameters=1)
        self.r = r

    # Implemented methods --------------------------------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealArray) -> RealArray:
        return -self.r * jnp.log1p(-jnp.exp(q[..., 0]))

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealArray) -> RealArray:
        return self.r / jnp.expm1(-q)

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealArray) -> RealArray:
        return -jnp.log1p(self.r / p)

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealArray) -> RealArray:
        return x[..., np.newaxis]

    # Overridden methods ---------------------------------------------------------------------------
    @overrides(ExponentialFamily)
    def carrier_measure(self, x: RealArray) -> RealArray:
        lgamma = jss.gammaln
        a = x + self.r - 1
        # Return log(a choose x).
        return lgamma(a + 1) - lgamma(x + 1) - lgamma(a - x + 1)

    @overrides(ExponentialFamily)
    def expected_carrier_measure(self, p: RealArray) -> RealArray:
        if self.r == 1:
            shape = p.shape[: -1]
            return jnp.zeros(shape)
        raise NotImplementedError

    # Magic methods --------------------------------------------------------------------------------
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NegativeBinomial):
            return NotImplemented
        return super().__eq__(other) and self.r == other.r

    def __hash__(self) -> int:
        return hash((self.num_parameters,
                     self.shape,
                     self.observation_shape,
                     self.r))


class Geometric(NegativeBinomial):

    def __init__(self) -> None:
        super().__init__(r=1)
