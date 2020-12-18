from typing import Any, Optional

import numpy as np
from jax import numpy as jnp
from jax.scipy import special as jss
from tjax import RealArray

from .exponential_family import ExponentialFamily
from .gamma import Gamma

__all__ = ['Poisson']


class Poisson(ExponentialFamily):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(num_parameters=1, **kwargs)

    # Implemented methods --------------------------------------------------------------------------
    def log_normalizer(self, q: RealArray) -> RealArray:
        # pylint: disable=no-self-use
        return jnp.exp(q)[..., 0]

    def nat_to_exp(self, q: RealArray) -> RealArray:
        return jnp.exp(q)

    def exp_to_nat(self, p: RealArray) -> RealArray:
        return jnp.log(p)

    def sufficient_statistics(self, x: RealArray) -> RealArray:
        return x[..., np.newaxis]

    # Overridden methods ---------------------------------------------------------------------------
    def carrier_measure(self, x: RealArray) -> RealArray:
        return -jss.gammaln(x + 1)

    def expected_carrier_measure(self, p: RealArray) -> RealArray:
        raise NotImplementedError

    def conjugate_prior_family(self) -> Optional[ExponentialFamily]:
        return Gamma(shape=self.shape)

    def conjugate_prior_distribution(self, p: RealArray, n: RealArray) -> RealArray:
        reshaped_n = n[..., np.newaxis]
        return jnp.append(-reshaped_n, reshaped_n * p, axis=-1)
