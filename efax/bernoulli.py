import jax.numpy as jnp
import jax.scipy.special as jss
from ipromise import overrides

from .multinomial import Multinomial
from .tensors import RealTensor

__all__ = ['Bernoulli']


class Bernoulli(Multinomial):

    def __init__(self, **kwargs):
        super().__init__(num_parameters=1, **kwargs)

    # Overridden methods ------------------------------------------------------
    @overrides(Multinomial)
    def log_normalizer(self, q: RealTensor) -> RealTensor:
        q = q[..., 0]
        return jnp.logaddexp(q, 0.0)

    @overrides(Multinomial)
    def nat_to_exp(self, q: RealTensor) -> RealTensor:
        return jss.expit(q)
