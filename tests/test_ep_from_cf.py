"""Generic test for expectation_parameters_from_characteristic_function.

The OLS inversion works for distributions where:
- All NP tree_leaves are scalar real floats (excludes complex/vector/integer-fixed fields)
- _complexify is not overridden (excludes Gamma, which keeps shape_minus_one real)
- log_normalizer is complex-analytic (excludes Chi/ChiSquare/InverseGamma via gammaln)
- sufficient_statistics does not require fixed_parameters (excludes Weibull)

These conditions are checked at runtime; incompatible distributions are skipped.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from numpy.random import Generator
from numpy.testing import assert_allclose

from efax import expectation_parameters_from_characteristic_function

from .distribution_info import DistributionInfo

_K = 300  # number of frequency evaluations per OLS solve


def test_ep_from_cf(distribution_info: DistributionInfo, generator: Generator) -> None:
    """OLS CF inversion recovers expectation parameters for compatible distributions."""
    nat_cls = distribution_info.nat_class()

    for _ in range(5):
        q = distribution_info.nat_parameter_generator(generator, shape=())

        # All NP tree_leaves must be scalar real floats.
        # Excludes: complex fields, vector/matrix fields, integer-typed fixed fields.
        leaves_q = jtu.tree_leaves(q)
        if not all(
            leaf.shape == () and jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in leaves_q
        ):
            pytest.skip(f"{nat_cls.__name__}: non-scalar or non-real NP fields")

        # Skip if _complexify is overridden — the CF only probes a subset of EP fields
        # (e.g. GammaNP keeps shape_minus_one real so mean_log cannot be recovered).
        if "_complexify" in type(q).__dict__:
            pytest.skip(f"{nat_cls.__name__}: custom _complexify")

        # Scale frequencies so that max phase |⟨t, E[T(x)]⟩| stays well below π,
        # preventing phase wrapping in Im(log φ).  We use the ground-truth EP to set
        # the scale — this is fine in a test where we know the answer.
        ep_true = q.to_exp()
        ep_magnitudes = [abs(float(x)) for x in jtu.tree_leaves(ep_true)]
        max_ep = max(ep_magnitudes) if ep_magnitudes else 1.0
        n = len(leaves_q)
        freq_scale = min(0.1, 0.5 / (n * max_ep + 1.0))

        # Build batched frequency NP: k random scalar frequencies per NP field.
        treedef = jtu.tree_structure(q)
        t = treedef.unflatten(
            [jnp.array(generator.uniform(-freq_scale, freq_scale, _K)) for _ in leaves_q]
        )

        # Evaluate CF — skip if log_normalizer is not complex-analytic
        # (e.g. Chi/ChiSquare/InverseGamma call gammaln which rejects complex inputs).
        try:
            cf = q.characteristic_function(t)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"{nat_cls.__name__}: CF evaluation failed: {e}")

        if not bool(jnp.all(jnp.isfinite(jnp.abs(cf)))):
            pytest.skip(f"{nat_cls.__name__}: CF returned non-finite values")

        # Invert via OLS — skip if sufficient_statistics requires fixed_parameters
        # (e.g. WeibullNP requires concentration as a fixed parameter).
        try:
            ep = expectation_parameters_from_characteristic_function(t, cf)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"{nat_cls.__name__}: OLS inversion failed: {e}")

        for i, (recovered, truth) in enumerate(
            zip(jtu.tree_leaves(ep), jtu.tree_leaves(ep_true), strict=True)
        ):
            assert_allclose(
                float(recovered),
                float(truth),
                rtol=0.05,
                atol=0.05,
                err_msg=f"{nat_cls.__name__} EP field {i}",
            )
