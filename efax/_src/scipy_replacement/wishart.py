from __future__ import annotations

import numpy as np
import scipy.stats as ss
from numpy.random import Generator
from tjax import NumpyRealArray, Shape
from typing import Any

from .shaped_distribution import ShapedDistribution


class ScipyWishart(ShapedDistribution[Any]):
    """A shape-aware wrapper around ``scipy.stats.wishart``."""

    def __init__(self, df: NumpyRealArray, scale: NumpyRealArray) -> None:
        shape = df.shape
        rvs_shape = scale.shape[-2:]
        dtype = scale.dtype
        objects = np.empty(shape, dtype=object)
        for i in np.ndindex(shape):
            objects[i] = ss.wishart(df=float(df[i]), scale=scale[i])
        super().__init__(shape, rvs_shape, dtype, objects, multivariate=True)

    def rvs(
        self, size: int | Shape | None = None, random_state: Generator | None = None
    ) -> NumpyRealArray:
        if size is None:
            size = ()
        elif isinstance(size, int):
            size = (size,)
        retval = np.empty(self.shape + size + self.rvs_shape, dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            this_object = self.access_object(i)
            if size:
                retval[i] = this_object.rvs(size=size, random_state=random_state)
            else:
                retval[i] = this_object.rvs(random_state=random_state)
        return retval
