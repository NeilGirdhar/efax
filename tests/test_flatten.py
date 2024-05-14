from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.random import Generator
from tjax import assert_tree_allclose

from .distribution_info import DistributionInfo


@pytest.mark.parametrize('natural', [False, True])
def test_flatten(generator: Generator,
                 distribution_info: DistributionInfo[Any, Any, Any],
                 *,
                 natural: bool,
                 ) -> None:
    """Test that the methods produce the correct shapes."""
    shape = (3, 4)
    p = (distribution_info.exp_parameter_generator(generator, shape=shape)
         if natural
         else distribution_info.nat_parameter_generator(generator, shape=shape))
    p_kwargs = p.fixed_parameters()
    p_flat = p.flattened()
    assert issubclass(p_flat.dtype.type, np.floating)
    assert_tree_allclose(type(p).unflattened(p_flat, **p_kwargs), p)
