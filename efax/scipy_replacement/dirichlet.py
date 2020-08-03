from typing import Optional

import numpy as np
from numpy.random import Generator
from scipy import stats as ss
from tjax import Shape

__all__ = ['ScipyDirichlet']


# pylint: disable=protected-access
class ScipyDirichletFixShape(ss._multivariate.dirichlet_frozen):

    def rvs(self, size: Shape = (), random_state: Optional[Generator] = None) -> np.ndarray:
        # This somehow fixes the behaviour of rvs.
        return super().rvs(size=size, random_state=random_state)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return super().pdf(x.T)
        return super().pdf(x)


class ScipyDirichlet:

    def __init__(self, parameters: np.ndarray):
        self.parameters = parameters
        self.component_shape = (parameters.shape[-1],)
        self.shape = parameters[..., -1].shape
        self.objects = np.empty(self.shape, dtype=object)
        for i in np.ndindex(self.shape):
            self.objects[i] = ScipyDirichletFixShape(parameters[i])

    def rvs(self, size: Shape = None, random_state: Generator = None) -> np.ndarray:
        if size is None:
            size = ()
        elif isinstance(size, int):
            size = (size,)
        retval = np.empty(self.shape + size + self.component_shape,
                          dtype=self.parameters.dtype)
        for i in np.ndindex(self.shape):
            retval[i] = self.objects[i].rvs(size=size,
                                            random_state=random_state)
        return retval

    def pdf(self, x: np.ndarray) -> np.ndarray:
        retval = np.empty(self.shape, dtype=self.parameters.dtype)
        for i in np.ndindex(self.shape):
            xi = x[i].astype(np.float64)
            if not np.allclose(1, np.sum(xi), atol=1e-5, rtol=0):
                raise ValueError
            if not np.allclose(1, np.sum(xi), atol=1e-10, rtol=0):
                xi /= np.sum(xi)
            retval[i] = self.objects[i].pdf(xi)
        return retval

    def entropy(self) -> np.ndarray:
        retval = np.empty(self.shape, dtype=self.parameters.dtype)
        for i in np.ndindex(self.shape):
            retval[i] = self.objects[i].entropy()
        return retval
