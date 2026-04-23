from __future__ import annotations

import jax.numpy as jnp
from numpy.testing import assert_allclose

from efax import CircularBoundedSupport, ComplexField


def test_complex_field_zero_plane_mapping_is_finite() -> None:
    field = ComplexField()
    flattened = field.flattened(jnp.asarray([0.0 + 0.0j]), map_to_plane=True)
    unflattened = field.unflattened(jnp.zeros(2), map_from_plane=True)

    assert_allclose(flattened, jnp.zeros(2))
    assert_allclose(unflattened, jnp.asarray([0.0 + 0.0j]))


def test_bounded_complex_field_zero_plane_mapping_is_finite() -> None:
    field = ComplexField(maximum_modulus=1.0)
    flattened = field.flattened(jnp.asarray([0.0 + 0.0j]), map_to_plane=True)
    unflattened = field.unflattened(jnp.zeros(2), map_from_plane=True)

    assert_allclose(flattened, jnp.zeros(2))
    assert_allclose(unflattened, jnp.asarray([0.0 + 0.0j]))


def test_circular_bounded_support_zero_plane_mapping_is_finite() -> None:
    support = CircularBoundedSupport(1.0)
    flattened = support.flattened(jnp.zeros(2), map_to_plane=True)
    unflattened = support.unflattened(jnp.zeros(2), 2, map_from_plane=True)

    assert_allclose(flattened, jnp.zeros(2))
    assert_allclose(unflattened, jnp.zeros(2))
