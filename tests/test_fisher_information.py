from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.random import Generator
from tjax import assert_tree_allclose

from efax import MultivariateNormalNP

from .distribution_info import DistributionInfo


def test_mvn_fisher_information() -> None:
    m = MultivariateNormalNP(jnp.zeros(2), -0.5 * jnp.eye(2))

    # pylint: disable=protected-access
    assert_tree_allclose(m._fisher_information_matrix(len(m.shape)),
                         np.array([[1., 0., 0., 0., 0.],
                                   [0., 1., 0., 0., 0.],
                                   [0., 0., 2., 0., 0.],
                                   [0., 0., 0., 4., 0.],
                                   [0., 0., 0., 0., 2.]]))

    assert_tree_allclose(
        m.fisher_information_diagonal(),
        MultivariateNormalNP(np.array([1.0, 1.0]),
                             np.array([[2., 4.], [4., 2.]])))

    assert_tree_allclose(m.fisher_information_trace(),
                         MultivariateNormalNP(2.0, 8.0))  # type: ignore


def test_mvn_fisher_information_b() -> None:
    m = MultivariateNormalNP(np.array([3.0, 5.0]), np.array([[-0.5, 0.0], [0.0, -0.8]]))

    # pylint: disable=protected-access
    assert_tree_allclose(m._fisher_information_matrix(len(m.shape)),
                         np.array([[1., -0., 6., 6.25, -0.],
                                   [-0., 0.625, -0., 3.75, 3.90625],
                                   [6., 0., 38., 37.5, 0.],
                                   [6.25, 3.75, 37.5, 64.062, 23.4375],
                                   [0., 3.90625, 0., 23.4375, 25.1953]]))

    assert_tree_allclose(
        m.fisher_information_diagonal(),
        MultivariateNormalNP(np.array([1.0, 0.625]),
                             np.array([[38., 64.062], [64.062, 25.1953]])))

    assert_tree_allclose(m.fisher_information_trace(),
                         MultivariateNormalNP(1.625, 127.258))  # type: ignore


def test_fisher_information_is_convex(generator: Generator,
                                      distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    shape = (3, 2)
    nat_parameters = distribution_info.nat_parameter_generator(generator, shape=shape)
    # pylint: disable=protected-access
    fisher_information = nat_parameters._fisher_information_matrix(len(shape))
    assert jnp.issubdtype(fisher_information.dtype, jnp.floating)
    eigvals = jnp.linalg.eigvals(fisher_information)
    if not jnp.all(eigvals >= 0.0):
        raise AssertionError(
            f"The Fisher information of {nat_parameters} is not convex.  Its eigenvalues are:"
            f"{eigvals}")
    determinant = jnp.linalg.det(fisher_information)
    assert determinant.shape == shape
    assert jnp.issubdtype(determinant.dtype, jnp.floating)

    if not jnp.all(determinant >= 0.0):
        raise AssertionError(
            f"The determinant of the Fisher information of {nat_parameters} is not all "
            f"nonnegative: {determinant}")
