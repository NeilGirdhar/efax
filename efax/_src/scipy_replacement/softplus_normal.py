from __future__ import annotations

import numpy as np
import scipy.stats as ss
from numpy.random import Generator
from tjax import (NumpyComplexArray, NumpyRealArray, NumpyRealNumeric, ShapeLike, inverse_softplus,
                  softplus)


class ScipySoftplusNormal:
    def __init__(self, mu: NumpyRealArray, sigma: NumpyRealArray) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def pdf(self, x: NumpyRealNumeric, out: None = None) -> NumpyRealArray:
        # Compute the inverse Softplus transformation: z = inverse_softplus(x)
        z = inverse_softplus(np.asarray(x))
        # Compute the normal PDF of z ~ N(mu, sigma^2)
        normal_pdf = ss.norm.pdf(z, loc=self.mu, scale=self.sigma)
        # Compute the Jacobian term dx/dz = 1/(1+exp(-x))
        jacobian = -1 / np.expm1(-x)
        # Compute the final PDF
        return normal_pdf * jacobian

    def rvs(self,
            size: ShapeLike | None = None,
            random_state: Generator | None = None
            ) -> NumpyComplexArray:
        distribution = ss.norm(loc=self.mu, scale=self.sigma)
        samples = np.asarray(distribution.rvs(size=size, random_state=random_state))
        return softplus(samples)
