from typing import Any

from jax import grad
from jax import numpy as jnp
from jax import vmap
from jax.tree_util import tree_map
from numpy.random import Generator
from tjax import assert_jax_allclose

from efax import HasConjugatePrior

from .create_info import VonMisesFisherInfo
from .distribution_info import DistributionInfo


def test_conjugate_prior(generator: Generator,
                         distribution_info: DistributionInfo[Any, Any]) -> None:
    """
    Test that the conjugate prior actually matches the distribution.
    """
    if isinstance(distribution_info, VonMisesFisherInfo):
        return

    shape = (4, 3) if distribution_info.supports_shape() else ()
    n = 100 * jnp.ones(shape)

    # Choose a random distribution.
    p = distribution_info.exp_parameter_generator(generator, shape=shape)

    if not isinstance(p, HasConjugatePrior):
        return

    # Find its conjugate prior at that point with many observations.
    cp_q = p.conjugate_prior_distribution(n)

    assert cp_q.shape() == p.shape()

    # Produce a copy of p that matches the conjugate prior distribution.
    cp_x = p.conjugate_prior_observation()

    # Produce a function that calculates the gradient of the density with respect to p.  Ensure that
    # it is broadcasted according to the shape.
    density_gradient = grad(type(cp_q).pdf, argnums=1)
    for _ in range(len(shape)):
        density_gradient = vmap(density_gradient)

    # Check the gradient of the density of the conjugate prior at p is zero.
    derivative = density_gradient(cp_q, cp_x)
    zero_derivative = tree_map(jnp.zeros_like, derivative)
    assert_jax_allclose(derivative, zero_derivative, atol=1.5)
