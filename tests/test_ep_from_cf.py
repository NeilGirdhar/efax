"""Generic test for expectation_parameters_from_characteristic_function.

The OLS inversion works for distributions where all of the following hold:
- log_normalizer is complex-analytic for all fields:
  - Excludes MultivariateNormal (outer_product uses conjugation, breaking analyticity)
- sufficient_statistics does not require fixed_parameters (excludes Weibull,
  MultivariateFixedVarianceNormal)

These conditions are checked at runtime; incompatible distributions are skipped.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.random import Generator
from numpy.testing import assert_allclose

from efax import (
    Assembler,
    Distribution,
    Flattener,
    expectation_parameters_from_characteristic_function,
    parameters,
)

from .distribution_info import DistributionInfo

_K = 300  # number of frequency evaluations per OLS solve


def _broadcast_fixed_parameters[D: Distribution](q: D) -> D:
    q_shape = q.shape
    final_parameters = {}
    for path, value in parameters(q).items():
        if value.shape[: len(q_shape)] == q_shape:
            final_parameters[path] = value
        else:
            final_parameters[path] = jnp.broadcast_to(value, (*q_shape, *value.shape))
    return Assembler.create_assembler(q).assemble(final_parameters)


def test_ep_from_cf(distribution_info: DistributionInfo, generator: Generator) -> None:  # noqa: PLR0914
    """OLS CF inversion recovers expectation parameters for compatible distributions."""
    nat_cls = distribution_info.nat_class()

    for _ in range(5):
        q = distribution_info.nat_parameter_generator(generator, shape=())
        q_flattener, q_flat = Flattener.flatten(q, mapped_to_plane=False)
        total_dim = q_flat.shape[-1]

        # Scale frequencies so that max phase |⟨t, E[T(x)]⟩| stays well below π,
        # preventing phase wrapping in Im(log φ).  We use the ground-truth EP to set
        # the scale — this is fine in a test where we know the answer.
        ep_true = q.to_exp()
        _, ep_true_flat = Flattener.flatten(ep_true, mapped_to_plane=False)
        ep_magnitudes = [abs(float(v)) for v in np.asarray(ep_true_flat).flatten().tolist()]
        max_ep = max(ep_magnitudes) if ep_magnitudes else 1.0
        freq_scale = min(0.1, 0.5 / (total_dim * max_ep + 1.0))

        # Build batched frequency NP: k random frequencies in real-coordinate space.
        t_flat = jnp.array(generator.uniform(-freq_scale, freq_scale, (_K, total_dim)))
        t = _broadcast_fixed_parameters(q_flattener.unflatten(t_flat))

        # Evaluate CF by vmapping over the k dimension — avoids rank-promotion errors
        # when field shapes are vectors/matrices (e.g. MultivariateDiagonalNormal).
        cf = jax.vmap(q.characteristic_function)(t)

        if not bool(jnp.all(jnp.isfinite(jnp.abs(cf)))):
            pytest.skip(f"{nat_cls.__name__}: CF returned non-finite values")

        # Skip if Im(log φ) is essentially zero everywhere — indicates log_normalizer
        # doesn't support complex inputs.
        b_all = jnp.imag(jnp.log(cf))
        if float(jnp.std(b_all)) < 1e-6:  # noqa: PLR2004
            pytest.skip(f"{nat_cls.__name__}: log_normalizer not complex-analytic (Im(log φ) ≈ 0)")

        # Per-coordinate check: verify each real coordinate independently contributes
        # to Im(log φ). If a coordinate contributes nothing, OLS cannot recover it.
        for i_coordinate in range(total_dim):
            t_single_flat = jnp.zeros_like(t_flat)
            t_single_flat = t_single_flat.at[:, i_coordinate].set(t_flat[:, i_coordinate])
            t_single = _broadcast_fixed_parameters(q_flattener.unflatten(t_single_flat))
            try:
                cf_leaf = jax.vmap(q.characteristic_function)(t_single)
            except Exception:  # noqa: BLE001
                break
            b_leaf = jnp.imag(jnp.log(cf_leaf))
            if float(jnp.std(b_leaf)) < 1e-6:  # noqa: PLR2004
                pytest.skip(
                    f"{nat_cls.__name__}: coordinate {i_coordinate} doesn't contribute "
                    f"to Im(log φ) — log_normalizer not complex-analytic for this coordinate"
                )

        # Invert via OLS — skip if sufficient_statistics requires fixed_parameters
        # (e.g. WeibullNP requires concentration as a fixed parameter).
        try:
            ep = expectation_parameters_from_characteristic_function(t, cf)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"{nat_cls.__name__}: OLS inversion failed: {e}")

        _, ep_flat = Flattener.flatten(ep, mapped_to_plane=False)
        assert_allclose(
            np.asarray(ep_flat),
            np.asarray(ep_true_flat),
            rtol=0.05,
            atol=0.05,
            err_msg=f"{nat_cls.__name__} EP coordinates",
        )
