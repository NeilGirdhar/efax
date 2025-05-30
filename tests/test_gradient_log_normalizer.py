from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from array_api_compat import array_namespace
from jax import grad, jvp, vjp
from jax.custom_derivatives import zero_from_primal
from numpy.random import Generator
from numpy.testing import assert_allclose
from tjax import JaxRealArray, assert_tree_allclose, jit

from efax import NaturalParametrization, Structure, parameters

from .distribution_info import DistributionInfo

_LogNormalizer: TypeAlias = Callable[[NaturalParametrization[Any, Any]], JaxRealArray]


def _prelude(generator: Generator,
             distribution_info: DistributionInfo[Any, Any, Any]
             ) -> tuple[_LogNormalizer, _LogNormalizer]:
    cls = distribution_info.nat_class()
    original_ln = cls._original_log_normalizer
    optimized_ln = cls.log_normalizer
    return original_ln, optimized_ln


def test_primals(generator: Generator, distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    """Tests that the gradient log-normalizer equals the gradient of the log-normalizer."""
    original_ln, optimized_ln = _prelude(generator, distribution_info)
    original_gln = jit(grad(original_ln, allow_int=True))
    optimized_gln = jit(grad(optimized_ln, allow_int=True))
    for _ in range(20):
        generated_np = distribution_info.nat_parameter_generator(generator, shape=())
        generated_ep = generated_np.to_exp()  # Regular transformation.
        generated_parameters = parameters(generated_ep, fixed=False)
        structure_ep = Structure.create(generated_ep)

        # Original GLN.
        original_gln_np = original_gln(generated_np)
        original_gln_ep = structure_ep.reinterpret(original_gln_np)
        original_gln_parameters = parameters(original_gln_ep, fixed=False)

        # Optimized GLN.
        optimized_gln_np = optimized_gln(generated_np)
        optimized_gln_ep = structure_ep.reinterpret(optimized_gln_np)
        optimized_gln_parameters = parameters(optimized_gln_ep, fixed=False)

        # Test primal evaluation.
        # parameters(generated_ep, fixed=False)
        assert_tree_allclose(generated_parameters, original_gln_parameters, rtol=1e-5)
        assert_tree_allclose(generated_parameters, optimized_gln_parameters, rtol=1e-5)


def _unit_tangent(nat_parameters: NaturalParametrization[Any, Any]
                 ) -> NaturalParametrization[Any, Any]:
    xp = array_namespace(nat_parameters)
    new_variable_parameters = {path: xp.ones_like(value)
                               for path, value in parameters(nat_parameters, fixed=False).items()}
    new_fixed_parameters = {path: zero_from_primal(value, symbolic_zeros=False)
                            for path, value in parameters(nat_parameters, fixed=True).items()}
    structure = Structure.create(nat_parameters)
    return structure.assemble({**new_variable_parameters, **new_fixed_parameters})


def test_jvp(generator: Generator, distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    """Tests that the gradient log-normalizer equals the gradient of the log-normalizer."""
    original_ln, optimized_ln = _prelude(generator, distribution_info)
    for _ in range(20):
        nat_parameters = distribution_info.nat_parameter_generator(generator, shape=())

        # Test JVP.
        nat_tangent = _unit_tangent(nat_parameters)
        original_ln_of_nat, original_jvp = jvp(original_ln, (nat_parameters,), (nat_tangent,))
        optimized_ln_of_nat, optimized_jvp = jvp(optimized_ln, (nat_parameters,), (nat_tangent,))
        assert_allclose(original_ln_of_nat, optimized_ln_of_nat, rtol=1.5e-5)
        assert_allclose(original_jvp, optimized_jvp, rtol=1.5e-5)


def test_vjp(generator: Generator, distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    """Tests that the gradient log-normalizer equals the gradient of the log-normalizer."""
    original_ln, optimized_ln = _prelude(generator, distribution_info)
    for _ in range(20):
        nat_parameters = distribution_info.nat_parameter_generator(generator, shape=())
        nat_tangent = _unit_tangent(nat_parameters)
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
