from __future__ import annotations

import jax.numpy as jnp
import pytest
from tjax import assert_tree_allclose

from efax import MultivariateFixedVarianceNormalNP, MultivariateNormalNP, NormalNP, parameter_concat

from .create_info import MultivariateFixedVarianceNormalInfo, MultivariateNormalInfo, NormalInfo


def test_normal_concat(distribution_name: str | None) -> None:
    NormalInfo.skip_if_deselected(distribution_name)
    p = NormalNP(jnp.asarray([1.0, 2.0]), jnp.asarray([-0.5, -1.0]))
    q = NormalNP(jnp.asarray([3.0]), jnp.asarray([-2.0]))

    expected = NormalNP(jnp.asarray([1.0, 2.0, 3.0]), jnp.asarray([-0.5, -1.0, -2.0]))
    assert_tree_allclose(parameter_concat([p, q]), expected)


def test_multivariate_concat_negative_axis(distribution_name: str | None) -> None:
    """Concatenate along the last axis of the distribution's shape, not of the parameters."""
    MultivariateNormalInfo.skip_if_deselected(distribution_name)
    dimensions = 3
    p = MultivariateNormalNP(
        jnp.ones((2, 1, dimensions)), -jnp.ones((2, 1, dimensions, dimensions))
    )
    q = MultivariateNormalNP(
        2.0 * jnp.ones((2, 4, dimensions)), -2.0 * jnp.ones((2, 4, dimensions, dimensions))
    )

    result = parameter_concat([p, q], axis=-1)

    assert result.shape == (2, 5)
    assert_tree_allclose(result[:, :1], p)
    assert_tree_allclose(result[:, 1:], q)


def test_fixed_parameters_are_concatenated(distribution_name: str | None) -> None:
    MultivariateFixedVarianceNormalInfo.skip_if_deselected(distribution_name)
    p = MultivariateFixedVarianceNormalNP(jnp.ones((1, 2)), variance=jnp.asarray([0.5]))
    q = MultivariateFixedVarianceNormalNP(jnp.zeros((2, 2)), variance=jnp.asarray([2.0, 3.0]))

    result = parameter_concat([p, q])

    assert_tree_allclose(result.variance, jnp.asarray([0.5, 2.0, 3.0]))
    assert_tree_allclose(
        result.mean_times_precision, jnp.asarray([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    )


def test_axis_out_of_bounds(distribution_name: str | None) -> None:
    NormalInfo.skip_if_deselected(distribution_name)
    p = NormalNP(jnp.asarray([1.0]), jnp.asarray([-0.5]))

    with pytest.raises(ValueError, match="out of bounds"):
        parameter_concat([p, p], axis=1)
