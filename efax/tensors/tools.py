import jax.numpy as jnp

__all__ = ['inverse_softplus']


def inverse_softplus(y):
    return jnp.where(y > 80.0,
                     y,
                     jnp.log(jnp.expm1(y)))
