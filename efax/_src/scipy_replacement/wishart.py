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

    def sample(self, shape: Shape = (), *, rng: Generator | None = None) -> NumpyRealArray:
        retval = np.empty(self.shape + shape + self.rvs_shape, dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            this_object = self.access_object(i)
            if shape:
                retval[i] = this_object.rvs(size=shape, random_state=rng)
            else:
                retval[i] = this_object.rvs(random_state=rng)
        return retval
