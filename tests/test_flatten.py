from typing import Any

from numpy.random import Generator
from tjax import assert_jax_allclose

from .distribution_info import DistributionInfo


def test_flatten(generator: Generator, distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    """
    Test that the methods produce the correct shapes.
    """
    shape = (3, 4) if distribution_info.supports_shape() else ()

    p = distribution_info.exp_parameter_generator(generator, shape=shape)
    q = distribution_info.nat_parameter_generator(generator, shape=shape)

    p_kwargs = p.fixed_parameters_mapping()
    q_kwargs = q.fixed_parameters_mapping()

    assert_jax_allclose(type(p).unflattened(p.flattened(), **p_kwargs), p)
    assert_jax_allclose(type(q).unflattened(q.flattened(), **q_kwargs), q)
