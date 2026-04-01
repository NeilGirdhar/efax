from __future__ import annotations

from dataclasses import dataclass

import array_api_extra as xpx
import numpy as np
from numpy.random import Generator
from tjax import NumpyIntegralArray, NumpyRealArray, Shape


@dataclass
class ScipyCategorical:
    probabilities: NumpyRealArray

    def pmf(self, x: NumpyIntegralArray) -> NumpyRealArray:
        return np.sum(x * self.probabilities, axis=-1)

    def sample(self, shape: Shape = (), *, rng: Generator | None = None) -> NumpyIntegralArray:
        rng = np.random.default_rng() if rng is None else rng
        cdf = np.cumsum(self.probabilities, axis=-1)
        uniforms = rng.random(shape + self.probabilities.shape[:-1])
        categories = np.sum(uniforms[..., np.newaxis] > cdf, axis=-1)
        return xpx.one_hot(categories, self.probabilities.shape[-1], dtype=np.int64)

    def entropy(self) -> NumpyRealArray:
        return -np.sum(self.probabilities * np.log(self.probabilities), axis=-1)
