from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.random import Generator
from tjax import assert_tree_allclose

from efax import Flattener, MultivariateUnitVarianceNormalNP

from .distribution_info import DistributionInfo


@pytest.mark.parametrize('natural', [False, True])
def test_flatten(generator: Generator,
                 distribution_info: DistributionInfo[Any, Any, Any],
                 *,
                 natural: bool,
                 ) -> None:
    """Test that unflattening restores the source of flattening."""
    shape = (3, 4)
    p = (distribution_info.exp_parameter_generator(generator, shape=shape)
         if natural
         else distribution_info.nat_parameter_generator(generator, shape=shape))
    flattener, flattened = Flattener.flatten(p)
    assert issubclass(flattened.dtype.type, np.floating)
    assert_tree_allclose(flattener.unflatten(flattened), p)


@pytest.mark.nondistribution
def test_raises() -> None:
    """Test that unflattening raises for bad shapes."""
    m = MultivariateUnitVarianceNormalNP(jnp.zeros(10))
    flattener, flattened = Flattener.flatten(m)
    assert flattened.shape == (10,)
    with pytest.raises(ValueError, match="Incompatible array"):
        flattener.unflatten(jnp.zeros(5))
    with pytest.raises(ValueError, match="Incompatible array"):
        flattener.unflatten(jnp.zeros(12))
