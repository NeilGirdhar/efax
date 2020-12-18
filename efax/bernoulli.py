from typing import Any, Optional

import numpy as np
from ipromise import overrides
from jax import numpy as jnp
from jax.scipy import special as jss
from tjax import RealArray

from .dirichlet import Beta
from .exponential_family import ExponentialFamily
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

    # Overridden methods ---------------------------------------------------------------------------
    def conjugate_prior_family(self) -> Optional[ExponentialFamily]:
        return Beta(shape=self.shape)

    def conjugate_prior_distribution(self, p: RealArray, n: RealArray) -> RealArray:
        """
        Args:
            p: The expectation parameters of a distribution having shape
                self.shape_including_parameters().
            n: The pseudo-observed count having shape self.shape.
        Returns: The natural parameters of the conjugate prior distribution.
        """
        if p.shape != self.shape_including_parameters():
            print(self.shape_including_parameters())
            raise ValueError
        if n.shape != self.shape:
            raise ValueError
        reshaped_n = n[..., np.newaxis]
        return reshaped_n * jnp.append(p, (1.0 - p), axis=-1)
