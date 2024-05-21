from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import jax.numpy as jnp
from jax import grad, jvp, vjp
from numpy.random import Generator
from numpy.testing import assert_allclose
from tjax import JaxRealArray, assert_tree_allclose, jit, zero_tangent_like

from efax import NaturalParametrization, Structure, parameters

from .create_info import BetaInfo, DirichletInfo, GammaInfo, GeneralizedDirichletInfo
from .distribution_info import DistributionInfo

LogNormalizer: TypeAlias = Callable[[NaturalParametrization[Any, Any]], JaxRealArray]


def test_conversion(generator: Generator,
                    distribution_info: DistributionInfo[Any, Any, Any]
                    ) -> None:
    """Test that the conversion between the different parametrizations are consistent."""
    atol = (5e-3
            if isinstance(distribution_info, GammaInfo | BetaInfo)
            else 2e-2
            if isinstance(distribution_info, DirichletInfo | GeneralizedDirichletInfo)
            else 1e-4)

    for _ in range(10):
        shape = (3, 4)
        original_ep = distribution_info.exp_parameter_generator(generator, shape=shape)
        intermediate_np = original_ep.to_nat()
        final_ep = intermediate_np.to_exp()

        # Check round trip.
        assert_tree_allclose(final_ep, original_ep, atol=atol, rtol=1e-4)

        # Check fixed parameters.
        original_fixed = parameters(original_ep, fixed=True)
        intermediate_fixed = parameters(intermediate_np, fixed=True)
        final_fixed = parameters(final_ep, fixed=True)
        assert_tree_allclose(original_fixed, intermediate_fixed)
        assert_tree_allclose(original_fixed, final_fixed)


def prelude(generator: Generator,
            distribution_info: DistributionInfo[Any, Any, Any]
            ) -> tuple[LogNormalizer, LogNormalizer]:
    cls = type(distribution_info.nat_parameter_generator(generator, shape=()))
    original_ln = cls._original_log_normalizer
    optimized_ln = cls.log_normalizer
    return original_ln, optimized_ln


def test_gradient_log_normalizer_primals(generator: Generator,
                                         distribution_info: DistributionInfo[Any, Any, Any]
                                         ) -> None:
    """Tests that the gradient log-normalizer equals the gradient of the log-normalizer."""
    original_ln, optimized_ln = prelude(generator, distribution_info)
    original_gln = jit(grad(original_ln, allow_int=True))
    optimized_gln = jit(grad(optimized_ln, allow_int=True))
    for _ in range(20):
        nat_parameters = distribution_info.nat_parameter_generator(generator, shape=())
        exp_parameters = nat_parameters.to_exp()  # Regular transformation.
        exp_structure = Structure.create(exp_parameters)
        # Able to unflatten into expectation parameters.

        # Original GLN.
        original_nat_parameters = original_gln(nat_parameters)
        original_exp_parameters = exp_structure.reinterpret(original_nat_parameters)

        # Optimized GLN.
        optimized_nat_parameters = optimized_gln(nat_parameters)
        optimized_exp_parameters = exp_structure.reinterpret(optimized_nat_parameters)

        # Test primal evaluation.
        assert_tree_allclose(exp_parameters, original_exp_parameters, rtol=1e-5)
        assert_tree_allclose(exp_parameters, optimized_exp_parameters, rtol=1e-5)


def unit_tangent(nat_parameters: NaturalParametrization[Any, Any]
                 ) -> NaturalParametrization[Any, Any]:
    new_variable_parameters = {path: jnp.ones_like(value)
                               for path, value in parameters(nat_parameters, fixed=False).items()}
    new_fixed_parameters = {path: zero_tangent_like(value)
                            for path, value in parameters(nat_parameters, fixed=True).items()}
    structure = Structure.create(nat_parameters)
    return structure.assemble({**new_variable_parameters, **new_fixed_parameters})


def test_gradient_log_normalizer_jvp(generator: Generator,
                                     distribution_info: DistributionInfo[Any, Any, Any]
                                     ) -> None:
    """Tests that the gradient log-normalizer equals the gradient of the log-normalizer."""
    original_ln, optimized_ln = prelude(generator, distribution_info)
    for _ in range(20):
        nat_parameters = distribution_info.nat_parameter_generator(generator, shape=())

        # Test JVP.
        nat_tangent = unit_tangent(nat_parameters)
        original_ln_of_nat, original_jvp = jvp(original_ln, (nat_parameters,), (nat_tangent,))
        optimized_ln_of_nat, optimized_jvp = jvp(optimized_ln, (nat_parameters,), (nat_tangent,))
        assert_allclose(original_ln_of_nat, optimized_ln_of_nat, rtol=1.5e-5)
        assert_allclose(original_jvp, optimized_jvp, rtol=1.5e-5)


def test_gradient_log_normalizer_vjp(generator: Generator,
                                     distribution_info: DistributionInfo[Any, Any, Any]
                                     ) -> None:
    """Tests that the gradient log-normalizer equals the gradient of the log-normalizer."""
    original_ln, optimized_ln = prelude(generator, distribution_info)
    for _ in range(20):
        nat_parameters = distribution_info.nat_parameter_generator(generator, shape=())
        nat_tangent = unit_tangent(nat_parameters)
        original_ln_of_nat, _ = jvp(original_ln, (nat_parameters,), (nat_tangent,))
        original_ln_of_nat_b, original_vjp = vjp(original_ln, nat_parameters)
        original_gln_of_nat, = original_vjp(1.0)
        optimized_ln_of_nat_b, optimized_vjp = vjp(optimized_ln, nat_parameters)
        optimized_gln_of_nat, = optimized_vjp(1.0)

        assert_allclose(original_ln_of_nat_b, optimized_ln_of_nat_b, rtol=1e-5)
        assert_allclose(original_ln_of_nat, original_ln_of_nat_b, rtol=1e-5)
        assert_tree_allclose(parameters(original_gln_of_nat, fixed=False),
                             parameters(optimized_gln_of_nat, fixed=False),
                             rtol=1e-5)
