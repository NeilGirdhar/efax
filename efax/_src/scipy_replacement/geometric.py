from __future__ import annotations

import numpy as np
import scipy.special
import scipy.stats as ss
from numpy.random import Generator
from tjax import NumpyComplexArray, NumpyRealArray, NumpyRealNumeric, ShapeLike

scipy_beta = scipy.special.beta


class ScipyGeometric:
    """This class repairs Geometric."""
    def __init__(self, p: NumpyRealArray) -> None:
        super().__init__()
        self.geom = ss.geom(p)

    def pmf(self, z: NumpyRealNumeric, out: None = None) -> NumpyRealArray:
        return self.geom.pmf(z)  # pyright: ignore

    def rvs(self,
            size: ShapeLike | None = None,
            random_state: Generator | None = None
            ) -> NumpyComplexArray:
        return np.asarray(self.geom.rvs(size=size, random_state=random_state))

    def entropy(self) -> NumpyRealArray:
        return np.asarray(self.geom.entropy())
