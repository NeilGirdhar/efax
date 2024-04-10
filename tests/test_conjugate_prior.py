from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import grad, vmap
from numpy.random import Generator
from tjax import assert_tree_allclose

from efax import HasConjugatePrior, HasGeneralizedConjugatePrior

from .distribution_info import DistributionInfo


def test_conjugate_prior(generator: Generator,
                         cp_distribution_info: DistributionInfo[Any, Any, Any],
                         distribution_name: None | str) -> None:
    """Test that the conjugate prior actually matches the distribution."""
    cp_distribution_info.skip_if_deselected(distribution_name)
    shape = (4, 3)
    n = 100.0 * jnp.ones(shape)

    # Choose a random distribution.
    p = cp_distribution_info.exp_parameter_generator(generator, shape=shape)
    assert isinstance(p, HasConjugatePrior)

    # Find its conjugate prior at that point with many observations.
    cp_q = p.conjugate_prior_distribution(n)

    assert cp_q.shape == p.shape

    # Produce a copy of p that matches the conjugate prior distribution.
    cp_x = p.conjugate_prior_observation()

    # Produce a function that calculates the gradient of the density with respect to p.  Ensure that
    # it is broadcasted according to the shape.
    density_gradient = grad(type(cp_q).pdf, argnums=1)
    for _ in range(len(shape)):
        density_gradient = vmap(density_gradient)

    # Check the gradient of the density of the conjugate prior at p is zero.
    derivative = density_gradient(cp_q, cp_x)
    zero_derivative = jnp.zeros_like(derivative)
    assert_tree_allclose(derivative, zero_derivative, atol=1.5)


def test_generalized_conjugate_prior(generator: Generator,
                                     gcp_distribution_info: DistributionInfo[Any, Any, Any],
                                     distribution_name: None | str
                                     ) -> None:
    """Same as test_conjugate_prior, but with generalized_conjugate_prior_distribution."""
    gcp_distribution_info.skip_if_deselected(distribution_name)
    shape = (4, 3)

    # Choose a random distribution.
    p = gcp_distribution_info.exp_parameter_generator(generator, shape=shape)
    assert isinstance(p, HasGeneralizedConjugatePrior)

    # Find its conjugate prior at that point with many observations.
    n = 100.0 * jnp.ones((*shape, p.dimensions()))
    gcp_q = p.generalized_conjugate_prior_distribution(n)

    assert gcp_q.shape == p.shape

    # Produce a copy of p that matches the conjugate prior distribution.
    cp_x = p.conjugate_prior_observation()

    # Produce a function that calculates the gradient of the density with respect to p.  Ensure that
    # it is broadcasted according to the shape.
    density_gradient = grad(type(gcp_q).pdf, argnums=1)
    for _ in range(len(shape)):
        density_gradient = vmap(density_gradient)

    # Check the gradient of the density of the conjugate prior at p is zero.
    derivative = density_gradient(gcp_q, cp_x)
    zero_derivative = jnp.zeros_like(derivative)
    assert_tree_allclose(derivative, zero_derivative, atol=1.5)
