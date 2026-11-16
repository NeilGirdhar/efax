from __future__ import annotations

import jax.numpy as jnp
import pytest
from tjax import assert_tree_allclose

from efax import MultivariateNormalNP, parameter_mean

from .create_info import MultivariateNormalInfo

_DIMENSIONS = 3


def _distribution() -> MultivariateNormalNP:
    """Return multivariate normals of shape (2, 5) whose parameters differ in every entry."""
    mean_times_precision = jnp.arange(2 * 5 * _DIMENSIONS, dtype=jnp.float64).reshape(
        2, 5, _DIMENSIONS
    )
    negative_half_precision = -jnp.arange(
        2 * 5 * _DIMENSIONS * _DIMENSIONS, dtype=jnp.float64
    ).reshape(2, 5, _DIMENSIONS, _DIMENSIONS)
    return MultivariateNormalNP(mean_times_precision, negative_half_precision)


def test_negative_axis_indexes_shape(distribution_name: str | None) -> None:
    """Average over the last axis of the distribution's shape, not of the parameters."""
    MultivariateNormalInfo.skip_if_deselected(distribution_name)
    p = _distribution()

    result = parameter_mean(p, axis=-1)

    expected = MultivariateNormalNP(
        jnp.mean(p.mean_times_precision, axis=1), jnp.mean(p.negative_half_precision, axis=1)
    )
    assert result.shape == (2,)
    assert_tree_allclose(result, expected)


def test_no_axis_averages_whole_shape(distribution_name: str | None) -> None:
    MultivariateNormalInfo.skip_if_deselected(distribution_name)
    p = _distribution()

    result = parameter_mean(p)

    expected = MultivariateNormalNP(
        jnp.mean(p.mean_times_precision, axis=(0, 1)),
        jnp.mean(p.negative_half_precision, axis=(0, 1)),
    )
    assert result.shape == ()
    assert_tree_allclose(result, expected)
    assert_tree_allclose(parameter_mean(p, axis=(0, -1)), expected)


def test_axis_out_of_bounds(distribution_name: str | None) -> None:
    MultivariateNormalInfo.skip_if_deselected(distribution_name)

    with pytest.raises(ValueError, match="out of bounds"):
        parameter_mean(_distribution(), axis=2)
