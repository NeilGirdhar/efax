from functools import partial

import jax.numpy as jnp
from jax import vjp
from numpy.random import Generator
from tjax import KeyArray

from .create_info import NormalInfo


def test_normal(distribution_name: str | None,
                generator: Generator, key: KeyArray
                ) -> None:
    NormalInfo.skip_if_deselected(distribution_name)
    distribution_info = NormalInfo()
    shape = (3,)
    n_samples = 4
    normal_np = distribution_info.nat_parameter_generator(generator, shape=shape)
    d = normal_np.to_deviation_parametrization()
    samples, f_vjp = vjp(partial(type(d).sample, key=key, shape=(n_samples,)), d)
    d_bar, = f_vjp(jnp.ones_like(samples))
    epsilon = (samples - d.mean) / d.deviation
    assert jnp.all(jnp.isclose(d_bar.mean, jnp.ones(shape) * float(n_samples)))
    assert jnp.all(jnp.isclose(d_bar.deviation, jnp.sum(epsilon, axis=0)))
