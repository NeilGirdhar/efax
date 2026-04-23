from __future__ import annotations

import jax.numpy as jnp
from jax import grad, vmap
from numpy.random import Generator
from numpy.testing import assert_allclose
from tjax import assert_tree_allclose

from efax import (
    BernoulliEP,
    Estimator,
    HasConjugatePrior,
    HasGeneralizedConjugatePrior,
    JointDistributionE,
    JointDistributionN,
    UnitVarianceNormalEP,
)

from .distribution_info import DistributionInfo


def test_conjugate_prior(
    generator: Generator, cp_distribution_info: DistributionInfo, distribution_name: str | None
) -> None:
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
    cp_x = p.as_conjugate_prior_observation()

    # Produce a function that calculates the gradient of the density with respect to p.  Ensure that
    # it is broadcasted according to the shape.
    density_gradient = grad(type(cp_q).pdf, argnums=1)
    for _ in range(len(shape)):
        density_gradient = vmap(density_gradient)

    # Check the gradient of the density of the conjugate prior at p is zero.
    derivative = density_gradient(cp_q, cp_x)
    zero_derivative = jnp.zeros_like(derivative)
    assert_tree_allclose(derivative, zero_derivative, atol=1.5)


def test_from_conjugate_prior(
    generator: Generator, cp_distribution_info: DistributionInfo, distribution_name: str | None
) -> None:
    """Test that the conjugate prior is reversible."""
    cp_distribution_info.skip_if_deselected(distribution_name)
    shape = (4, 3)
    n = 123.0 * jnp.ones(shape)

    # Choose a random distribution.
    p = cp_distribution_info.exp_parameter_generator(generator, shape=shape)
    assert isinstance(p, HasConjugatePrior)

    # Find its conjugate prior at that point with many observations.
    cp_q = p.conjugate_prior_distribution(n)
    p_estimator = Estimator.from_expectation(p)
    p_prime, n_prime = p_estimator.from_conjugate_prior(cp_q)
    assert_tree_allclose(p, p_prime, atol=1.5)
    assert_allclose(n, n_prime, atol=1.5)


def test_joint_from_conjugate_prior() -> None:
    """Test that joint CP recovery passes each leaf its own CP distribution."""
    n = 123.0 * jnp.ones((4, 3))
    p = JointDistributionE(
        {
            "normal": UnitVarianceNormalEP(jnp.full((4, 3), 0.25)),
            "bernoulli": BernoulliEP(jnp.full((4, 3), 0.7)),
        }
    )
    cp_sub_distributions = {}
    for name, value in p.sub_distributions().items():
        assert isinstance(value, HasConjugatePrior)
        cp_sub_distributions[name] = value.conjugate_prior_distribution(n)
    cp_q = JointDistributionN(cp_sub_distributions)

    p_prime, n_prime = Estimator.from_expectation(p).from_conjugate_prior(cp_q)

    assert_tree_allclose(p, p_prime)
    assert_allclose(n, n_prime)


def test_generalized_conjugate_prior(
    generator: Generator, gcp_distribution_info: DistributionInfo, distribution_name: str | None
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
    cp_x = p.as_conjugate_prior_observation()

    # Produce a function that calculates the gradient of the density with respect to p.  Ensure that
    # it is broadcasted according to the shape.
    density_gradient = grad(type(gcp_q).pdf, argnums=1)
    for _ in range(len(shape)):
        density_gradient = vmap(density_gradient)

    # Check the gradient of the density of the conjugate prior at p is zero.
    derivative = density_gradient(gcp_q, cp_x)
    zero_derivative = jnp.zeros_like(derivative)
    assert_tree_allclose(derivative, zero_derivative, atol=1.5)
