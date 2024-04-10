from math import isclose

import jax.numpy as jnp
import pytest
from jax import grad, vmap
from jax.random import split
from tjax import KeyArray

from efax import random_gamma


@pytest.mark.parametrize('a', [1e3, 1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3])
def test_gamma_generator(key: KeyArray, a: float) -> None:
    n = 100000
    gg = grad(random_gamma, argnums=1)
    vgg = vmap(gg, (0, None))
    keys = split(key, (n,))
    grads = vgg(keys, a)
    mg = jnp.mean(grads)
    assert isclose(mg, 1.0, rel_tol=1e-2)
