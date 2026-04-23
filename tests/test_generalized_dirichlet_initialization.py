from __future__ import annotations

import jax.numpy as jnp
from numpy.random import default_rng
from tjax import assert_tree_allclose

from tests.create_info import GeneralizedDirichletInfo


def test_generalized_dirichlet_initial_search_parameters_are_finite() -> None:
    info = GeneralizedDirichletInfo(dimensions=5)
    p = info.exp_parameter_generator(default_rng(123), shape=(3, 4))

    initial = p.initial_search_parameters()

    assert initial.shape == (3, 4, 10)
    assert bool(jnp.all(jnp.isfinite(initial)))


def test_generalized_dirichlet_to_nat_round_trip() -> None:
    info = GeneralizedDirichletInfo(dimensions=5)
    q = info.nat_parameter_generator(default_rng(123), shape=(3, 4))

    assert_tree_allclose(q.to_exp().to_nat(), q, rtol=1e-4, atol=1e-4)
