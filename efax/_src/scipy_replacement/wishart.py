from __future__ import annotations

import numpy as np
import scipy.stats as ss
from tjax import NumpyRealArray
from typing import override

from .shaped_distribution import ShapedDistribution


class ScipyWishart(ShapedDistribution[object]):
    """This class allows distributions having a non-empty shape."""
    @override
    def __init__(self, df: NumpyRealArray, scale: NumpyRealArray) -> None:
        shape = df.shape
        rvs_shape = scale.shape
        dtype = df.dtype
        objects = np.empty(shape, dtype=object)
        for i in np.ndindex(shape):
            objects[i] = ss.wishart(df=df[i], scale=scale[i])
        super().__init__(shape, rvs_shape, dtype, objects)
