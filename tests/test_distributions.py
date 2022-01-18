from typing import Any

import jax.numpy as jnp
from jax import grad, jit, jvp, vjp
from numpy.random import Generator
from numpy.testing import assert_allclose
from tjax import assert_tree_allclose, zero_tangent_like

from .create_info import BetaInfo, DirichletInfo, GammaInfo
from .distribution_info import DistributionInfo


def test_conversion(generator: Generator,
                    distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    """
    Test that the conversion between the different parametrizations are consistent.
    """
    atol = (5e-3
            if isinstance(distribution_info, (GammaInfo, BetaInfo))
            else 2e-2
            if isinstance(distribution_info, DirichletInfo)
            else 1e-4)

    for _ in range(10):
        shape = (3, 4)
        original_ep = distribution_info.exp_parameter_generator(generator, shape=shape)
        intermediate_np = original_ep.to_nat()
        final_ep = intermediate_np.to_exp()
        try:
            assert_tree_allclose(final_ep, original_ep, atol=atol, rtol=1e-4)
            original_fixed = original_ep.fixed_parameters_mapping()
            intermediate_fixed = intermediate_np.fixed_parameters_mapping()
            final_fixed = final_ep.fixed_parameters_mapping()
            assert original_fixed.keys() == intermediate_fixed.keys() == final_fixed.keys()
            for name, value in original_fixed.items():
                assert_allclose(value, intermediate_fixed[name])
                assert_allclose(value, final_fixed[name])
        except AssertionError:
            print(original_ep, intermediate_np, final_ep)
            raise


def test_gradient_log_normalizer(generator: Generator,
                                 distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    """
    Tests that the gradient log-normalizer evaluates to the same as the gradient of the
    log-normalizer.
    """
    # pylint: disable=too-many-locals, disable=protected-access
    cls = type(distribution_info.nat_parameter_generator(generator, shape=()))
    original_ln = cls._original_log_normalizer
    original_gln = jit(grad(cls._original_log_normalizer, allow_int=True))
    optimized_ln = cls.log_normalizer
    optimized_gln = jit(grad(optimized_ln, allow_int=True))

    for _ in range(20):
        nat_parameters = distribution_info.nat_parameter_generator(generator, shape=())
        kw_nat_parameters = nat_parameters.fixed_parameters_mapping()
        exp_parameters = nat_parameters.to_exp()  # Regular transformation.
        nat_cls = type(nat_parameters)
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
        assert_tree_allclose(exp_parameters, original_exp_parameters, rtol=1e-5)
        assert_tree_allclose(exp_parameters, optimized_exp_parameters, rtol=1e-5)

        # Test JVP.
        ones_like_nat_parameters = nat_cls(
            **{name: zero_tangent_like(value)
               for name, value in nat_parameters.fixed_parameters_mapping().items()},
            **{name: jnp.ones_like(value)
               for name, value in nat_parameters.parameters_name_value()})
        original_ln_of_nat, original_jvp = jvp(original_ln, (nat_parameters,),
                                               (ones_like_nat_parameters,))
        optimized_ln_of_nat, optimized_jvp = jvp(optimized_ln, (nat_parameters,),
                                                 (ones_like_nat_parameters,))
        assert_allclose(original_ln_of_nat, optimized_ln_of_nat, rtol=1.5e-5)
        assert_allclose(original_jvp, optimized_jvp, rtol=1.5e-5)

        # Test VJP.
        original_ln_of_nat_b, original_vjp = vjp(original_ln, nat_parameters)
        original_gln_of_nat, = original_vjp(1.0)
        optimized_ln_of_nat_b, optimized_vjp = vjp(optimized_ln, nat_parameters)
        optimized_gln_of_nat, = optimized_vjp(1.0)

        assert_allclose(original_ln_of_nat_b, optimized_ln_of_nat_b, rtol=1e-5)
        assert_allclose(original_ln_of_nat, original_ln_of_nat_b, rtol=1e-5)

        for name, original_value in original_gln_of_nat.parameters_name_value():
            optimized_value = getattr(optimized_gln_of_nat, name)
            assert_allclose(original_value, optimized_value, rtol=1e-5)
