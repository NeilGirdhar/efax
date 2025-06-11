"""These tests verify entropy gradients."""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import pytest
from jax import grad, jit, tree, value_and_grad
from jax.test_util import check_grads
from numpy.random import Generator
from rich.console import Console
from tjax import (JaxBooleanArray, JaxRealArray, assert_tree_allclose, dynamic_tree_all,
                  print_generic)

from efax import Flattener, GammaEP, GammaVP, HasEntropy

from .create_info import GammaInfo
from .distribution_info import DistributionInfo


@jit
def _sum_entropy(flattened: JaxRealArray, flattener: Flattener) -> JaxRealArray:
    x = flattener.unflatten(flattened)
    return jnp.sum(x.entropy())


@jit
def _all_finite(some_tree: object, /) -> JaxBooleanArray:
    return dynamic_tree_all(tree.map(lambda x: jnp.all(jnp.isfinite(x)), some_tree))


def _check_entropy_gradient(distribution: HasEntropy, /) -> None:
    flattener, flattened = Flattener.flatten(distribution, map_to_plane=False)
    p_sum_entropy = partial(_sum_entropy, flattener=flattener)
    calculated_gradient = grad(p_sum_entropy)(flattened)
    if not _all_finite(calculated_gradient):
        indices = jnp.argwhere(jnp.isnan(calculated_gradient))
        bad_distributions = [distribution[tuple(index)[:-1]] for index in indices]
        console = Console()
        with console.capture() as capture:
            print_generic(bad_distributions, console=console)
        msg = f"Non-finite gradient found for distributions: {capture.get()}"
        pytest.fail(msg)
    check_grads(p_sum_entropy, (flattened,), order=1, atol=1e-4, rtol=1e-2)


def test_nat_entropy_gradient(generator: Generator,
                              entropy_distribution_info: DistributionInfo,
                              ) -> None:
    shape = (7, 13)
    nat_parameters = entropy_distribution_info.nat_parameter_generator(generator, shape=shape)
    _check_entropy_gradient(nat_parameters)


def test_exp_entropy_gradient(generator: Generator,
                              entropy_distribution_info: DistributionInfo,
                              ) -> None:
    shape = (7, 13)
    exp_parameters = entropy_distribution_info.exp_parameter_generator(generator, shape=shape)
    _check_entropy_gradient(exp_parameters)


def test_gamma_vp_entropy_gradient(distribution_name: str | None) -> None:
    GammaInfo.skip_if_deselected(distribution_name)

    def f(flattened: JaxRealArray, flattener: Flattener[GammaEP]) -> JaxRealArray:
        x_e = flattener.unflatten(flattened)
        return x_e.entropy()

    def g(flattened: JaxRealArray, flattener: Flattener[GammaEP]) -> JaxRealArray:
        x_e = flattener.unflatten(flattened)
        x_n = x_e.to_nat()
        return x_n.entropy()

    o = jnp.ones(())
    x = GammaVP(o, o * 0.25)
    x_e = x.to_exp()
    flattener, flattened = Flattener.flatten(x_e, map_to_plane=False)
    f_value, f_gradient = value_and_grad(f)(flattened, flattener)
    g_value, g_gradient = value_and_grad(g)(flattened, flattener)
    assert_tree_allclose(f_value, g_value)
    assert_tree_allclose(f_gradient, g_gradient)
