from jax import grad, jit, jvp
from jax import numpy as jnp
from jax import vjp
from numpy.random import Generator
from numpy.testing import assert_allclose

from efax import VonMises

from .distribution_info import DistributionInfo

# todo: Block VonMises until https://github.com/google/jax/issues/2466 is
# resolved.


def test_conversion(generator: Generator, distribution_info: DistributionInfo) -> None:
    """
    Test that the conversion between the different parametrizations are consistent.
    """
    if isinstance(distribution_info.exp_family, VonMises):
        return
    exp_family = distribution_info.exp_family

    for _ in range(10):
        parameters = distribution_info.exp_parameter_generator(generator, shape=(4, 3))
        nat_parameters = exp_family.exp_to_nat(parameters)
        exp_parameters = exp_family.nat_to_exp(nat_parameters)
        assert_allclose(parameters, exp_parameters, rtol=1e-4)


def test_gradient_log_normalizer(generator: Generator, distribution_info: DistributionInfo) -> None:
    """
    Tests that the gradient log-normalizer evaluates to the same as the gradient of the
    log-normalizer.
    """
    if isinstance(distribution_info.exp_family, VonMises):
        return
    exp_family = distribution_info.exp_family

    # pylint: disable=protected-access
    original_f = exp_family._original_log_normalizer

    f = exp_family.log_normalizer

    original_gln = jit(grad(original_f))
    gln = jit(grad(f))
    nat_to_exp = exp_family.nat_to_exp

    for _ in range(20):
        nat_parameters = distribution_info.nat_parameter_generator(generator, shape=())

        original_gln_x = original_gln(nat_parameters)
        gln_x = gln(nat_parameters)
        exp_parameters = nat_to_exp(nat_parameters)

        # Test primal evaluation.
        assert_allclose(original_gln_x, exp_parameters, rtol=1e-5)
        assert_allclose(gln_x, exp_parameters, rtol=1e-5)

        # Test JVP.
        original_gradients = jvp(original_f, (nat_parameters,), (jnp.ones_like(nat_parameters),))
        gradients = jvp(f, (nat_parameters,), (jnp.ones_like(nat_parameters),))
        assert_allclose(original_gradients, gradients, rtol=1e-5)

        # Test VJP.
        _, original_g = vjp(original_f, nat_parameters)
        _, g = vjp(f, nat_parameters)
        assert_allclose(original_g(1.0), g(1.0), rtol=1e-5)
