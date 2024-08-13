"""These tests are related to entropy."""
from __future__ import annotations

from operator import add
from typing import Any

import jax.numpy as jnp
import pytest
from jax import grad, tree, value_and_grad
from numpy.random import Generator
from tjax import JaxRealArray, assert_tree_allclose

from efax import GammaEP, GammaVP, HasEntropyEP, HasEntropyNP

from .create_info import BetaInfo, DirichletInfo
from .distribution_info import DistributionInfo


def total_infinite(some_tree: Any, /) -> bool:
    return tree.reduce(add, tree.map(lambda x: x.size - jnp.sum(jnp.isfinite(x)), some_tree), 0)


def sum_entropy(x: HasEntropyEP[Any] | HasEntropyNP[Any], /) -> JaxRealArray:
    return jnp.sum(x.entropy())


def all_finite(some_tree: Any, /) -> bool:
    return tree.all(tree.map(lambda x: jnp.all(jnp.isfinite(x)), some_tree))


def check_entropy_gradient(distribution: HasEntropyEP[Any] | HasEntropyNP[Any], /) -> None:
    calculated_gradient = grad(sum_entropy)(distribution)
    if not all_finite(calculated_gradient):
        print(type(distribution), total_infinite(calculated_gradient))  # noqa: T201
    assert all_finite(calculated_gradient)


def test_nat_entropy_gradient(generator: Generator,
                              entropy_distribution_info: DistributionInfo[Any, Any, Any],
                              ) -> None:
    if isinstance(entropy_distribution_info, BetaInfo | DirichletInfo):
        pytest.skip()
    shape = (7, 13)
    nat_parameters = entropy_distribution_info.nat_parameter_generator(generator, shape=shape)
    check_entropy_gradient(nat_parameters)


def test_exp_entropy_gradient(generator: Generator,
                              entropy_distribution_info: DistributionInfo[Any, Any, Any],
                              ) -> None:
    if isinstance(entropy_distribution_info, BetaInfo | DirichletInfo):
        pytest.skip()
    shape = (7, 13)
    exp_parameters = entropy_distribution_info.exp_parameter_generator(generator, shape=shape)
    check_entropy_gradient(exp_parameters)


def test_gamma_vp_entropy_gradient() -> None:
    def f(x_e: GammaEP) -> JaxRealArray:
        return x_e.entropy()

    def g(x_e: GammaEP) -> JaxRealArray:
        x_n = x_e.to_nat()
        return x_n.entropy()

    o = jnp.ones(())
    x = GammaVP(o, o * 0.25)
    x_e = x.to_exp()
    f_value, f_gradient = value_and_grad(f)(x_e)
    g_value, g_gradient = value_and_grad(g)(x_e)
    assert_tree_allclose(f_value, g_value)
    assert_tree_allclose(f_gradient, g_gradient)
