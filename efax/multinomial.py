import jax.numpy as jnp
import jax.scipy.special as jss
import numpy as np
from ipromise import implements

from .exponential_family import ExponentialFamily
from .tensors import RealTensor

__all__ = ['Multinomial']


class Multinomial(ExponentialFamily):

    def __init__(self, *, num_parameters: int, **kwargs):
        if num_parameters < 0:
            raise ValueError
        if num_parameters == 1:
            observation_shape = ()
        else:
            observation_shape = (num_parameters,)
        super().__init__(num_parameters=num_parameters,
                         observation_shape=observation_shape,
                         **kwargs)

    # Magic methods -----------------------------------------------------------
    def __repr__(self):
        return (f"{type(self).__name__}(shape={self.shape}, "
                f"num_parameters={self.num_parameters})")

    # Implemented methods -----------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealTensor) -> RealTensor:
        max_q = jnp.maximum(0.0, jnp.amax(q, axis=-1))
        q_minus_max_q = q - max_q[..., np.newaxis]
        log_scaled_A = jnp.logaddexp(
            -max_q, jss.logsumexp(q_minus_max_q, axis=-1))
        return max_q + log_scaled_A

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealTensor) -> RealTensor:
        max_q = jnp.maximum(0.0, jnp.amax(q, axis=-1))
        q_minus_max_q = q - max_q[..., np.newaxis]
        log_scaled_A = jnp.logaddexp(
            -max_q, jss.logsumexp(q_minus_max_q, axis=-1))
        return jnp.exp(q_minus_max_q
                       - log_scaled_A[..., np.newaxis])

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealTensor) -> RealTensor:
        p_k = 1.0 - jnp.sum(p, axis=-1, keepdims=True)
        return jnp.log(p / p_k)

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealTensor) -> RealTensor:
        return x

    # New methods -------------------------------------------------------------
    @staticmethod
    def nat_to_probability(q: RealTensor) -> RealTensor:
        max_q = jnp.maximum(0.0, jnp.amax(q, axis=-1))
        q_minus_max_q = q - max_q[..., np.newaxis]
        log_scaled_A = jnp.logaddexp(
            -max_q, jss.logsumexp(q_minus_max_q, axis=-1))
        p = jnp.exp(q_minus_max_q
                    - log_scaled_A[..., np.newaxis])
        final_p = 1.0 - jnp.sum(p, axis=-1, keepdims=True)
        return jnp.concatenate([p, final_p], axis=-1)

    @staticmethod
    def nat_to_surprisal(q: RealTensor) -> RealTensor:
        total_p = Multinomial.nat_to_probability(q)
        return -jnp.log(total_p)
