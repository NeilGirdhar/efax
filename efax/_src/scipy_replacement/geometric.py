from __future__ import annotations

import numpy as np
import scipy.stats as ss
from numpy.random import Generator
from tjax import NumpyComplexArray, NumpyRealArray, NumpyRealNumeric, ShapeLike


class ScipyGeometric:
    """This class repairs Geometric."""
    def __init__(self, p: NumpyRealArray) -> None:
        super().__init__()
        self.geom = ss.geom(p)

    def pmf(self, z: NumpyRealNumeric, out: None = None) -> NumpyRealArray:
        return np.asarray(self.geom.pmf(z))

    def rvs(self,
            size: ShapeLike | None = None,
            random_state: Generator | None = None
            ) -> NumpyComplexArray:
        return np.asarray(self.geom.rvs(size=size, random_state=random_state))

    def entropy(self) -> NumpyRealArray:
        return np.asarray(self.geom.entropy())
