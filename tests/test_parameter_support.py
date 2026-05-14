from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from efax import (
    CircularBoundedSupport,
    ComplexField,
    HermitianMatrixSupport,
    SimplexSupport,
    SubsimplexSupport,
    SymmetricMatrixSupport,
)


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


def test_symmetric_matrix_support_is_transpose_symmetric() -> None:
    support = SymmetricMatrixSupport(ring=ComplexField())
    matrix = jnp.asarray(
        [
            [1 + 0j, 2 + 3j],
            [2 + 3j, 4 + 0j],
        ]
    )

    flattened = support.flattened(matrix, map_to_plane=False)
    unflattened = support.unflattened(flattened, 2, map_from_plane=False)

    expected_flattened = jnp.asarray([1, 2, 4, 0, 3, 0])

    assert support.num_elements(2) == expected_flattened.shape[-1]
    assert_allclose(flattened, expected_flattened)
    assert_allclose(unflattened, matrix)


def test_hermitian_matrix_support_round_trip() -> None:
    support = HermitianMatrixSupport()
    matrix = jnp.asarray(
        [
            [1 + 0j, 2 + 3j],
            [2 - 3j, 4 + 0j],
        ]
    )

    flattened = support.flattened(matrix, map_to_plane=False)
    unflattened = support.unflattened(flattened, 2, map_from_plane=False)

    expected_flattened = jnp.asarray([1, 2, 4, 3])

    assert support.num_elements(2) == expected_flattened.shape[-1]
    assert_allclose(flattened, expected_flattened)
    assert_allclose(unflattened, matrix)


def test_hermitian_matrix_support_generation_is_hermitian() -> None:
    support = HermitianMatrixSupport(positive_semidefinite=True)
    matrix = support.generate(jnp, np.random.default_rng(123), (5,), 0.01, 3)

    assert matrix.shape == (5, 3, 3)
    assert_allclose(matrix, jnp.conj(jnp.matrix_transpose(matrix)), rtol=1e-6)
    assert bool(jnp.all(jnp.linalg.eigvalsh(matrix) >= 0.0))


def test_simplex_support_uses_reduced_dimension() -> None:
    support = SimplexSupport()
    dimensions = 4
    reduced_dimensions = dimensions - 1

    assert support.shape(dimensions) == (reduced_dimensions,)
    assert support.num_elements(dimensions) == reduced_dimensions
    assert_allclose(
        support.unflattened(jnp.asarray([0.2, 0.3, 0.4]), dimensions, map_from_plane=False),
        jnp.asarray([0.2, 0.3, 0.4]),
    )


def test_simplex_support_plane_round_trip() -> None:
    support = SimplexSupport()
    x = jnp.asarray([[0.2, 0.3, 0.1], [0.1, 0.1, 0.7]])

    flattened = support.flattened(x, map_to_plane=True)
    unflattened = support.unflattened(flattened, 4, map_from_plane=True)

    assert_allclose(unflattened, x, rtol=1e-6)


def test_simplex_support_generate() -> None:
    support = SimplexSupport()
    x = support.generate(jnp, np.random.default_rng(123), (5, 2), 0.01, 4)
    residual = 1.0 - jnp.sum(x, axis=-1)

    assert x.shape == (5, 2, 3)
    assert bool(jnp.all(x > 0.0))
    assert bool(jnp.all(residual > 0.0))
    assert bool(jnp.all(jnp.sum(x, axis=-1) < 1.0))


def test_subsimplex_support_preserves_dimension() -> None:
    support = SubsimplexSupport()
    dimensions = 4

    assert support.shape(dimensions) == (dimensions,)
    assert support.num_elements(dimensions) == dimensions
    assert_allclose(
        support.unflattened(jnp.asarray([0.2, 0.3, 0.1, 0.15]), dimensions, map_from_plane=False),
        jnp.asarray([0.2, 0.3, 0.1, 0.15]),
    )


def test_subsimplex_support_plane_round_trip() -> None:
    support = SubsimplexSupport()
    x = jnp.asarray([[0.2, 0.3, 0.1, 0.15], [0.1, 0.1, 0.2, 0.3]])

    flattened = support.flattened(x, map_to_plane=True)
    unflattened = support.unflattened(flattened, 4, map_from_plane=True)

    assert_allclose(unflattened, x, rtol=1e-6)


def test_subsimplex_support_clamp_preserves_subsimplex() -> None:
    support = SubsimplexSupport()
    clamped = support.clamp(jnp.asarray([[0.8, 0.7, 0.6], [0.2, 0.3, 0.4]]))

    assert bool(jnp.all(clamped > 0.0))
    assert bool(jnp.all(jnp.sum(clamped, axis=-1) < 1.0))
