from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.random import Generator
from tjax import assert_tree_allclose

from efax import MultivariateNormalNP

from .create_info import MultivariateNormalInfo
from .distribution_info import DistributionInfo


def test_mvn_fisher_information(distribution_name: str | None) -> None:
    MultivariateNormalInfo.skip_if_deselected(distribution_name)
    m = MultivariateNormalNP(jnp.zeros(2), -0.5 * jnp.eye(2))

    assert_tree_allclose(m._fisher_information_matrix(),  # noqa: SLF001
                         np.asarray([[1., 0., 0., 0., 0.],
                                     [0., 1., 0., 0., 0.],
                                     [0., 0., 2., 0., 0.],
                                     [0., 0., 0., 4., 0.],
                                     [0., 0., 0., 0., 2.]]))

    assert_tree_allclose(
        m.fisher_information_diagonal(),
        MultivariateNormalNP(jnp.asarray([1.0, 1.0]),
                             jnp.asarray([[2., 4.], [4., 2.]])))

    assert_tree_allclose(m.fisher_information_trace(),
                         MultivariateNormalNP(jnp.asarray(2.0), jnp.asarray(8.0)))


def test_mvn_fisher_information_b(distribution_name: str | None) -> None:
    MultivariateNormalInfo.skip_if_deselected(distribution_name)
    m = MultivariateNormalNP(jnp.asarray([3.0, 5.0]), jnp.asarray([[-0.5, 0.0], [0.0, -0.8]]))

    assert_tree_allclose(m._fisher_information_matrix(),  # noqa: SLF001
                         jnp.asarray([[1., -0., 6., 6.25, -0.],
                                      [-0., 0.625, -0., 3.75, 3.90625],
                                      [6., 0., 38., 37.5, 0.],
                                      [6.25, 3.75, 37.5, 64.062, 23.4375],
                                      [0., 3.90625, 0., 23.4375, 25.1953]]))

    assert_tree_allclose(
        m.fisher_information_diagonal(),
        MultivariateNormalNP(jnp.asarray([1.0, 0.625]),
                             jnp.asarray([[38., 64.062], [64.062, 25.1953]])))

    assert_tree_allclose(m.fisher_information_trace(),
                         MultivariateNormalNP(jnp.asarray(1.625), jnp.asarray(127.258)))


def test_fisher_information_is_convex(generator: Generator,
                                      distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    shape = (3, 2)
    nat_parameters = distribution_info.nat_parameter_generator(generator, shape=shape)
    fisher_information = nat_parameters._fisher_information_matrix()  # noqa: SLF001
    assert issubclass(fisher_information.dtype.type, jnp.floating)
    eigvals = jnp.linalg.eigvals(fisher_information)
    if not jnp.all(eigvals >= 0.0):
        msg = (f"The Fisher information of {nat_parameters} is not convex.  Its eigenvalues are:"
               f"{eigvals}")
        raise AssertionError(msg)
    determinant = jnp.linalg.det(fisher_information)
    assert determinant.shape == shape
    assert issubclass(determinant.dtype.type, jnp.floating)

    if not jnp.all(determinant >= 0.0):
        msg = (f"The determinant of the Fisher information of {nat_parameters} is not all "
               f"nonnegative: {determinant}")
        raise AssertionError(msg)
