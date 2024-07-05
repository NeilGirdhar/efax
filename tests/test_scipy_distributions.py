from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.random import Generator as NumpyGenerator
from tjax import Shape

from efax import ScipyDirichlet, ScipyMultivariateNormal


@pytest.mark.nondistribution
@pytest.mark.parametrize(('distribution', 'm', 'n'),
                         [(ScipyMultivariateNormal.from_mc(mean=np.zeros((3, 4))), (3,), 4),
                          (ScipyMultivariateNormal.from_mc(cov=np.eye(2)), (), 2),
                          (ScipyMultivariateNormal.from_mc(), (), 1),
                          (ScipyDirichlet(alpha=np.ones((2, 5))), (2,), 5),
                          (ScipyDirichlet(alpha=np.ones((2, 3, 5))), (2, 3), 5),
                          ])
def test_shaped(generator: NumpyGenerator, distribution: Any, m: Shape, n: int) -> None:
    assert distribution.rvs().shape == (*m, n)
    assert distribution.rvs(1).shape == (*m, 1, n)
    assert distribution.rvs((1,)).shape == (*m, 1, n)
    assert distribution.rvs((1, 1, 1)).shape == (*m, 1, 1, 1, n)
    assert distribution.rvs((1, 2, 3)).shape == (*m, 1, 2, 3, n)
