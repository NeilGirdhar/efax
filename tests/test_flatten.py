from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import Generator
from tjax import assert_tree_allclose

from efax import ExpectationParametrization, NaturalParametrization

from .distribution_info import DistributionInfo


def test_flatten(generator: Generator,
                 distribution_info: DistributionInfo[NaturalParametrization[Any, Any],
                                                     ExpectationParametrization[Any],
                                                     Any]) -> None:
    """Test that the methods produce the correct shapes."""
    shape = (3, 4)

    p = distribution_info.exp_parameter_generator(generator, shape=shape)
    q = distribution_info.nat_parameter_generator(generator, shape=shape)

    p_kwargs = p.fixed_parameters()
    q_kwargs = q.fixed_parameters()

    p_flat = p.flattened()
    q_flat = q.flattened()

    assert issubclass(p_flat.dtype.type, np.floating)
    assert issubclass(q_flat.dtype.type, np.floating)

    assert_tree_allclose(type(p).unflattened(p_flat, **p_kwargs), p)
    assert_tree_allclose(type(q).unflattened(q_flat, **q_kwargs), q)
