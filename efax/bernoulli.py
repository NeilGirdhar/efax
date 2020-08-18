from typing import Any

import numpy as np
from ipromise import overrides
from jax import numpy as jnp
from jax.scipy import special as jss
from tjax import RealArray

from .multinomial import Multinomial

__all__ = ['Bernoulli']


class Bernoulli(Multinomial):

    def __init__(self, **kwargs: Any):
        super().__init__(num_parameters=1, **kwargs)

    # Overridden methods ---------------------------------------------------------------------------
    @overrides(Multinomial)
    def log_normalizer(self, q: RealArray) -> RealArray:
        q = q[..., 0]
        return jnp.logaddexp(q, 0.0)

    @overrides(Multinomial)
    def nat_to_exp(self, q: RealArray) -> RealArray:
        return jss.expit(q)

    @overrides(Multinomial)
    def sufficient_statistics(self, x: RealArray) -> RealArray:
        return x[..., np.newaxis]
