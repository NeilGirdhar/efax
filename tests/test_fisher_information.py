import jax.numpy as jnp
from tjax import assert_tree_allclose

from efax import MultivariateNormalNP


def test_fisher_information() -> None:
    m = MultivariateNormalNP(jnp.zeros(2), -0.5 * jnp.eye(2))

    assert_tree_allclose(
        m.fisher_information(),
        MultivariateNormalNP(MultivariateNormalNP(jnp.array([[1., 0.], [0., 1.]]),  # type: ignore
                                                  jnp.array([[[0., 0.], [0., 0.]],
                                                             [[0., 0.], [0., 0.]]])),
                             MultivariateNormalNP(jnp.array([[[0., 0.], [0., 0.]],  # type: ignore
                                                             [[0., 0.], [0., 0.]]]),
                                                  jnp.array([[[[2., 0.], [0., 0.]],
                                                              [[0., 2.], [0., 0.]]],
                                                             [[[0., 0.], [2., 0.]],
                                                              [[0., 0.], [0., 2.]]]]))))

    assert_tree_allclose(
        m.fisher_information(diagonal=True),
        MultivariateNormalNP(jnp.array([1.0, 1.0]),
                             jnp.array([[2., 0.], [0., 2.]])))

    assert_tree_allclose(m.fisher_information(trace=True),
                         MultivariateNormalNP(2.0, 8.0))  # type: ignore
