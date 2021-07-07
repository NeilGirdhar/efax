from typing import Any, List

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from tjax import RealArray, Shape

__all__: List[str] = []


class ShapedDistribution:
    """
    Allow a distributions with shape.
    """
    def __init__(self,
                 shape: Shape,
                 component_shape: Shape,
                 dtype: np.dtype[Any],
                 objects: npt.NDArray[np.object_]):
        self.shape = shape
        self.component_shape = component_shape
        self.dtype = dtype
        self.objects = objects

    def rvs(self, size: Shape = None, random_state: Generator = None) -> RealArray:
        if size is None:
            size = ()
        elif isinstance(size, int):
            size = (size,)
        retval = np.empty(self.shape + size + self.component_shape,
                          dtype=self.dtype)
        for i in np.ndindex(*self.shape):
            retval[i] = self.objects[i].rvs(size=size,
                                            random_state=random_state)
        return retval

    def pdf(self, x: RealArray) -> RealArray:
        retval = np.empty(self.shape, dtype=self.dtype)
        for i in np.ndindex(*self.shape):
            retval[i] = self.objects[i].pdf(x[i])
        return retval

    def entropy(self) -> RealArray:
        retval = np.empty(self.shape, dtype=self.dtype)
        for i in np.ndindex(*self.shape):
            retval[i] = self.objects[i].entropy()
        return retval
