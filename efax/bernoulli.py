from typing import Any

from ipromise import overrides
from jax import numpy as jnp
from jax.scipy import special as jss
from tjax import RealTensor

from .multinomial import Multinomial

__all__ = ['Bernoulli']


class Bernoulli(Multinomial):

    def __init__(self, **kwargs: Any):
        super().__init__(num_parameters=1, **kwargs)

    # Overridden methods ---------------------------------------------------------------------------
    @overrides(Multinomial)
    def log_normalizer(self, q: RealTensor) -> RealTensor:
        q = q[..., 0]
        return jnp.logaddexp(q, 0.0)

    @overrides(Multinomial)
    def nat_to_exp(self, q: RealTensor) -> RealTensor:
        return jss.expit(q)
