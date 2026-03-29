from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator as NumpyGenerator
from tjax import Shape

from efax import ScipyDirichlet, ScipyDistribution, ScipyMultivariateNormal, ScipyWishart


@pytest.mark.nondistribution
@pytest.mark.parametrize(
    ("distribution", "m", "event_shape"),
    [
        (ScipyMultivariateNormal.from_mc(mean=np.zeros((3, 4))), (3,), (4,)),
        (ScipyMultivariateNormal.from_mc(cov=np.eye(2)), (), (2,)),
        (ScipyMultivariateNormal.from_mc(), (), (1,)),
        (ScipyDirichlet(alpha=np.ones((2, 5))), (2,), (5,)),
        (ScipyDirichlet(alpha=np.ones((2, 3, 5))), (2, 3), (5,)),
        (ScipyWishart(df=np.full((2,), 5.0), scale=np.tile(np.eye(3), (2, 1, 1))), (2,), (3, 3)),
    ],
)
def test_shaped(
    generator: NumpyGenerator, distribution: ScipyDistribution, m: Shape, event_shape: Shape
) -> None:
    assert distribution.rvs().shape == (*m, *event_shape)
    assert distribution.rvs(1).shape == (*m, 1, *event_shape)
    assert distribution.rvs((1,)).shape == (*m, 1, *event_shape)
    assert distribution.rvs((1, 1, 1)).shape == (*m, 1, 1, 1, *event_shape)
    assert distribution.rvs((1, 2, 3)).shape == (*m, 1, 2, 3, *event_shape)
