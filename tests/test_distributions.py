from typing import Any

import jax.numpy as jnp
from jax import grad, jit, jvp, vjp
from jax.tree_util import tree_map
from numpy.random import Generator
from numpy.testing import assert_allclose
from tjax import assert_jax_allclose, field_names_and_values

from .create_info import BetaInfo, DirichletInfo, GammaInfo, VonMisesFisherInfo
from .distribution_info import DistributionInfo

# TODO: Block VonMises until https://github.com/google/jax/issues/2466 is resolved.


def test_conversion(generator: Generator, distribution_info: DistributionInfo[Any, Any]) -> None:
    """
    Test that the conversion between the different parametrizations are consistent.
    """
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
    # TODO: Remove when https://github.com/tensorflow/probability/issues/1247 is resolved.
    if isinstance(distribution_info, VonMisesFisherInfo):
        return

    # pylint: disable=protected-access
    cls = type(distribution_info.nat_parameter_generator(generator, shape=()))
    original_f = cls._original_log_normalizer
    original_gln = jit(grad(cls._original_log_normalizer))
    optimized_gln = jit(grad(cls.log_normalizer))

    for _ in range(20):
        nat_parameters = distribution_info.nat_parameter_generator(generator, shape=())
        kw_nat_parameters = nat_parameters.unflattened_kwargs()
        exp_parameters = nat_parameters.to_exp()  # Regular transformation.
        ep_cls = type(exp_parameters)

        # Original GLN.
        original_nat_parameters = original_gln(nat_parameters)
        f_original_nat_parameters = original_nat_parameters.flattened()
        original_exp_parameters = ep_cls.unflattened(f_original_nat_parameters, **kw_nat_parameters)

        # Optimized GLN.
        optimized_nat_parameters = optimized_gln(nat_parameters)
        f_optimized_nat_parameters = optimized_nat_parameters.flattened()
        optimized_exp_parameters = ep_cls.unflattened(f_optimized_nat_parameters,
                                                      **kw_nat_parameters)

        # Test primal evaluation.
        assert_jax_allclose(exp_parameters, original_exp_parameters, rtol=1e-5)
        assert_jax_allclose(exp_parameters, optimized_exp_parameters, rtol=1e-5)

        # Test JVP.
        ones_like_nat_parameters = tree_map(jnp.ones_like, nat_parameters)
        original_gradients = jvp(original_f, (nat_parameters,), (ones_like_nat_parameters,))
        gradients = jvp(cls.log_normalizer, (nat_parameters,), (ones_like_nat_parameters,))
        assert_allclose(original_gradients, gradients, rtol=1e-5)

        # Test VJP.
        _, original_g = vjp(original_f, nat_parameters)
        _, g = vjp(cls.log_normalizer, nat_parameters)
        assert_jax_allclose(original_g(1.0)[0], g(1.0)[0], rtol=1e-5)
