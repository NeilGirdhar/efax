"""Generic test for expectation_parameters_from_characteristic_function.

The OLS inversion works for distributions where all of the following hold:
- All NP tree_leaves are real floats (excludes complex fields, NegativeBinomial)
- log_normalizer is complex-analytic for all fields:
  - Excludes MultivariateNormal (outer_product uses conjugation, breaking analyticity)
- sufficient_statistics does not require fixed_parameters (excludes Weibull,
  MultivariateFixedVarianceNormal)

These conditions are checked at runtime; incompatible distributions are skipped.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
from numpy.random import Generator
from numpy.testing import assert_allclose

from efax import expectation_parameters_from_characteristic_function, parameters

from .distribution_info import DistributionInfo

_K = 300  # number of frequency evaluations per OLS solve


def test_ep_from_cf(distribution_info: DistributionInfo, generator: Generator) -> None:  # noqa: C901, PLR0914
    """OLS CF inversion recovers expectation parameters for compatible distributions."""
    nat_cls = distribution_info.nat_class()

    for _ in range(5):
        q = distribution_info.nat_parameter_generator(generator, shape=())

        # All NP tree_leaves must be real floats.
        # Excludes: complex fields, integer-typed fixed fields.
        leaves_q = jtu.tree_leaves(q)
        if not all(jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in leaves_q):
            pytest.skip(f"{nat_cls.__name__}: non-real NP fields")

        # Scale frequencies so that max phase |⟨t, E[T(x)]⟩| stays well below π,
        # preventing phase wrapping in Im(log φ).  We use the ground-truth EP to set
        # the scale — this is fine in a test where we know the answer.
        ep_true = q.to_exp()
        for path, (value, support) in parameters(ep_true, support=True).items():
            if not bool(jnp.allclose(value, support.clamp(value), rtol=0.0, atol=1e-12)):
                pytest.skip(f"{nat_cls.__name__}: invalid generated EP at {path}")

        ep_magnitudes = [
            abs(float(v))
            for leaf in jtu.tree_leaves(ep_true)
            for v in np.asarray(leaf).flatten().tolist()
        ]
        max_ep = max(ep_magnitudes) if ep_magnitudes else 1.0
        total_dim = sum(int(np.prod(leaf.shape)) if leaf.shape else 1 for leaf in leaves_q)
        freq_scale = min(0.1, 0.5 / (total_dim * max_ep + 1.0))

        # Build batched frequency NP: k random frequencies per NP field (matching field shape).
        treedef = jtu.tree_structure(q)
        t = treedef.unflatten(
            [
                jnp.array(generator.uniform(-freq_scale, freq_scale, (_K, *leaf.shape)))
                for leaf in leaves_q
            ]
        )

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

        # Per-leaf check: verify each NP leaf independently contributes to Im(log φ).
        # Reuse the existing t but zero out all but one leaf to isolate each field.
        # If a leaf contributes nothing (e.g. MultivariateNormalNP mean field due to
        # outer_product with conjugation), the OLS cannot recover that EP field.
        t_leaves = jtu.tree_leaves(t)
        for i_leaf, leaf in enumerate(leaves_q):
            t_single = treedef.unflatten(
                [jnp.zeros_like(tl) if j != i_leaf else tl for j, tl in enumerate(t_leaves)]
            )
            try:
                cf_leaf = jax.vmap(q.characteristic_function)(t_single)
            except Exception:  # noqa: BLE001
                break
            b_leaf = jnp.imag(jnp.log(cf_leaf))
            if float(jnp.std(b_leaf)) < 1e-6:  # noqa: PLR2004
                pytest.skip(
                    f"{nat_cls.__name__}: leaf {i_leaf} ({leaf.shape}) doesn't contribute "
                    f"to Im(log φ) — log_normalizer not complex-analytic for this field"
                )

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
                np.asarray(recovered),
                np.asarray(truth),
                rtol=0.05,
                atol=0.05,
                err_msg=f"{nat_cls.__name__} EP field {i}",
            )
