from __future__ import annotations

import jax.numpy as jnp
from tjax import assert_tree_allclose

from efax import MultivariateFixedVarianceNormalNP, MultivariateNormalNP, NormalPP

from .create_info import MultivariateFixedVarianceNormalInfo, MultivariateNormalInfo, NormalInfo


def test_normal_scale_evidence(distribution_name: str | None) -> None:
    NormalInfo.skip_if_deselected(distribution_name)
    p = NormalPP(jnp.asarray([1.0, -2.0, 3.0]), jnp.asarray([0.5, 1.0, 4.0])).to_nat()
    scale = jnp.asarray([2.0, 0.5, 0.0])

    scaled = p.scale_evidence(scale).to_precision_parametrization()

    assert_tree_allclose(scaled.mean[:2], jnp.asarray([1.0, -2.0]))
    assert_tree_allclose(scaled.precision, jnp.asarray([1.0, 0.5, 0.0]))


def test_scalar_scale_evidence(distribution_name: str | None) -> None:
    NormalInfo.skip_if_deselected(distribution_name)
    p = NormalPP(jnp.asarray([1.0, -2.0]), jnp.asarray([0.5, 1.0])).to_nat()

    scaled = p.scale_evidence(jnp.asarray(3.0)).to_precision_parametrization()

    assert_tree_allclose(scaled.mean, jnp.asarray([1.0, -2.0]))
    assert_tree_allclose(scaled.precision, jnp.asarray([1.5, 3.0]))


def test_multivariate_scale_evidence(distribution_name: str | None) -> None:
    """Scale each component's vector and matrix parameters by that component's entry.

    The number of components equals the dimension, so a scale broadcast over the parameters'
    trailing axes would have the right shape but scale the wrong entries.
    """
    MultivariateNormalInfo.skip_if_deselected(distribution_name)
    mean_times_precision = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    negative_half_precision = jnp.asarray(
        [[[-1.0, 0.25], [0.25, -2.0]], [[-3.0, 0.5], [0.5, -4.0]]]
    )
    p = MultivariateNormalNP(mean_times_precision, negative_half_precision)
    scale = jnp.asarray([2.0, 10.0])

    expected = MultivariateNormalNP(
        jnp.asarray([[2.0, 4.0], [30.0, 40.0]]),
        jnp.asarray([[[-2.0, 0.5], [0.5, -4.0]], [[-30.0, 5.0], [5.0, -40.0]]]),
    )
    assert_tree_allclose(p.scale_evidence(scale), expected)


def test_fixed_parameters_are_not_scaled(distribution_name: str | None) -> None:
    MultivariateFixedVarianceNormalInfo.skip_if_deselected(distribution_name)
    p = MultivariateFixedVarianceNormalNP(
        jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), variance=jnp.asarray([0.5, 2.0])
    )

    scaled = p.scale_evidence(jnp.asarray([2.0, 10.0]))

    assert_tree_allclose(scaled.mean_times_precision, jnp.asarray([[2.0, 4.0], [30.0, 40.0]]))
    assert_tree_allclose(scaled.variance, jnp.asarray([0.5, 2.0]))
