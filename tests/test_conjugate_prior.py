from jax import grad
from jax import numpy as jnp
from jax import vmap
from numpy.random import Generator
from numpy.testing import assert_allclose

from efax import VonMises

from .distribution_info import DistributionInfo


def test_conjugate_prior(generator: Generator, distribution_info: DistributionInfo) -> None:
    """
    Test that the conjugate prior actually matches the distribution.
    """
    if isinstance(distribution_info.exp_family, VonMises):
        return

    shape = (4, 3)
    n = 100 * jnp.ones(shape)

    exp_family = distribution_info.with_shape(shape).exp_family
    cp_family = exp_family.conjugate_prior_family()

    if cp_family is None:
        return

    assert cp_family.shape == exp_family.shape

    # Choose a random distribution.
    p = distribution_info.exp_parameter_generator(generator, shape=shape)

    # Produce a copy of p that matches the conjugate prior distribution.
    cp_x = p.reshape(cp_family.shape_including_observations())

    # Find its conjugate prior at that point with many observations.
    cp_q = exp_family.conjugate_prior_distribution(p, n)

    # Check the gradient of the density of the conjugate prior at p is zero.
    density = vmap(vmap(grad(cp_family.pdf)))

    derivative = density(cp_q, cp_x)
    assert_allclose(derivative, 0.0, atol=1.5)
