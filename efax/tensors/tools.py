import jax.numpy as jnp

from .annotations import Tensor

__all__ = ['inverse_softplus']


def inverse_softplus(y: Tensor) -> Tensor:
    return jnp.where(y > 80.0,
                     y,
                     jnp.log(jnp.expm1(y)))
