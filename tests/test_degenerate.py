"""Degenerate behaviour."""
from functools import partial

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import vjp
from numpy.random import Generator
from tjax import KeyArray, print_generic

from efax import GammaNP


# gamma=GammaNP[dataclass]
# ├── negative_rate=Jax Array (2,) float32
# │   └──  -0.0492 │ -0.0458
# └── shape_minus_one=Jax Array (2,) float32
#     └──  1.6102 │ 1.5054
# key=Jax Array (2,) uint32
# └──  4093016152 │ 3163742808
# retval=tuple
# ├── GammaNP[dataclass]
# │   ├── negative_rate=Jax Array (2,) float32
# │   │   └──  6250383802368.0000 │ 1265401.1250
# │   └── shape_minus_one=Jax Array (2,) float32
# │       └──  1823734235136.0000 │ -147.8925
# └── Jax Array () bool
#     └── False
# sample_bar=Jax Array (2,) float32
# └──  3096.8057 │ 2845.2410
@pytest.mark.skip
def test_gamma_sampling_explosion(generator: Generator, key: KeyArray) -> None:
    d = GammaNP(jnp.asarray([-0.0492, -0.0458]),
                jnp.asarray([1.6102, 1.5054]))
    key = jr.wrap_key_data(jnp.asarray([4093016152, 3163742808], jnp.uint32))
    samples, f_vjp = vjp(partial(GammaNP.sample, key=key), d)
    d_bar, = f_vjp(jnp.asarray([3096.8057, 2845.2410]))
    print_generic(samples=samples, d_bar=d_bar)
