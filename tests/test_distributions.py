from typing import Any

from jax import grad, jit, jvp
from jax import numpy as jnp
from jax import vjp
from jax.tree_util import tree_map
from numpy.random import Generator
from numpy.testing import assert_allclose
from tjax import assert_jax_allclose, field_names_and_values, field_values

from .create_info import BetaInfo, DirichletInfo, GammaInfo, VonMisesFisherInfo
from .distribution_info import DistributionInfo

# TODO: Block VonMises until https://github.com/google/jax/issues/2466 is resolved.


def test_conversion(generator: Generator, distribution_info: DistributionInfo[Any, Any]) -> None:
    """
    Test that the conversion between the different parametrizations are consistent.
    """
    if isinstance(distribution_info, VonMisesFisherInfo):
        return

    atol = (5e-3
            if isinstance(distribution_info, (GammaInfo, BetaInfo))
            else 2e-2
            if isinstance(distribution_info, DirichletInfo)
            else 1e-4)

    for _ in range(10):
        shape = (3, 4) if distribution_info.supports_shape() else ()
        original_ep = distribution_info.exp_parameter_generator(generator, shape=shape)
        intermediate_np = original_ep.to_nat()
        final_ep = intermediate_np.to_exp()
        try:
            assert_jax_allclose(final_ep, original_ep, atol=atol, rtol=1e-4)
            assert (list(field_names_and_values(original_ep, static=True))
                    == list(field_names_and_values(intermediate_np, static=True))
                    == list(field_names_and_values(final_ep, static=True)))
        except AssertionError:
            print(original_ep, intermediate_np, final_ep)
            raise


def test_gradient_log_normalizer(generator: Generator,
                                 distribution_info: DistributionInfo[Any, Any]) -> None:
    """
    Tests that the gradient log-normalizer evaluates to the same as the gradient of the
    log-normalizer.
    """
    if isinstance(distribution_info, VonMisesFisherInfo):
        return

    # pylint: disable=protected-access
    cls = type(distribution_info.nat_parameter_generator(generator, shape=()))
    original_f = cls._original_log_normalizer
    f = cls.log_normalizer
    original_gln = jit(grad(cls._original_log_normalizer))
    gln = jit(grad(f))

    for _ in range(20):
        nat_parameters = distribution_info.nat_parameter_generator(generator, shape=())

        exp_parameters = nat_parameters.to_exp()
        ep_cls = type(exp_parameters)
        original_gln_x = ep_cls(*field_values(original_gln(nat_parameters)))
        gln_x = ep_cls(*field_values(gln(nat_parameters)))

        # Test primal evaluation.
        assert_jax_allclose(exp_parameters, original_gln_x, rtol=1e-5)
        assert_jax_allclose(exp_parameters, gln_x, rtol=1e-5)

        # Test JVP.
        ones_like_nat_parameters = tree_map(jnp.ones_like, nat_parameters)
        original_gradients = jvp(original_f, (nat_parameters,), (ones_like_nat_parameters,))
        gradients = jvp(f, (nat_parameters,), (ones_like_nat_parameters,))
        assert_allclose(original_gradients, gradients, rtol=1e-5)

        # Test VJP.
        _, original_g = vjp(original_f, nat_parameters)
        _, g = vjp(f, nat_parameters)
        assert_jax_allclose(original_g(1.0)[0], g(1.0)[0], rtol=1e-5)
