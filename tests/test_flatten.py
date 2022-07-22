from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import Generator
from tjax import assert_tree_allclose

from .distribution_info import DistributionInfo


def test_flatten(generator: Generator, distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    """
    Test that the methods produce the correct shapes.
    """
    shape = (3, 4)

    p = distribution_info.exp_parameter_generator(generator, shape=shape)
    q = distribution_info.nat_parameter_generator(generator, shape=shape)

    p_kwargs = p.fixed_parameters_mapping()
    q_kwargs = q.fixed_parameters_mapping()

    p_flat = p.flattened()
    q_flat = q.flattened()

    assert np.issubdtype(p_flat.dtype, np.floating)
    assert np.issubdtype(q_flat.dtype, np.floating)

    assert_tree_allclose(type(p).unflattened(p_flat, **p_kwargs), p)
    assert_tree_allclose(type(q).unflattened(q_flat, **q_kwargs), q)
