from __future__ import annotations

import numpy as np
import scipy.stats as ss
from numpy.random import Generator
from tjax import NumpyComplexArray, NumpyRealArray, NumpyRealNumeric, ShapeLike


class ScipyLogNormal:
    def __init__(self, mu: NumpyRealArray, sigma: NumpyRealArray) -> None:
        super().__init__()
        self.mu = mu
        self.distribution = ss.lognorm(s=sigma)

    def pdf(self, z: NumpyRealNumeric, out: None = None) -> NumpyRealArray:
        exp_mu = np.exp(self.mu)
        return np.asarray(self.distribution.pdf(z / exp_mu)) / exp_mu

    def rvs(self,
            size: ShapeLike | None = None,
            random_state: Generator | None = None
            ) -> NumpyComplexArray:
        return np.asarray(self.distribution.rvs(size=size, random_state=random_state)
                          * np.exp(self.mu))

    def entropy(self) -> NumpyRealArray:
        return np.asarray(self.distribution.entropy()) + self.mu
