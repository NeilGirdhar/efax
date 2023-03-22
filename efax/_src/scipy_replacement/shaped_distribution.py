from __future__ import annotations

from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from tjax import NumpyComplexArray, NumpyRealArray, Shape
from typing_extensions import override

__all__: list[str] = []


T = TypeVar('T')


class ShapedDistribution(Generic[T]):
    """Allow a distributions with shape."""
    @override
    def __init__(self,
                 shape: Shape,
                 rvs_shape: Shape,
                 rvs_dtype: np.dtype[Any],
                 objects: npt.NDArray[np.object_]):
        super().__init__()
        self.shape = shape
        self.rvs_shape = rvs_shape
        self.rvs_dtype = rvs_dtype
        self.real_dtype: np.dtype[Any] = np.zeros(0, dtype=rvs_dtype).real.dtype
        self.objects = objects

    def rvs(self,
            size: None | int | Shape = None,
            random_state: Generator | None = None) -> NumpyRealArray:
        if size is None:
            size = ()
        elif isinstance(size, int):
            size = (size,)
        retval = np.empty(self.shape + size + self.rvs_shape,
                          dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            retval[i] = self.objects[i].rvs(size=size, random_state=random_state)
        return retval

    def pdf(self, x: NumpyComplexArray) -> NumpyRealArray:
        retval = np.empty(self.shape, dtype=self.real_dtype)
        for i in np.ndindex(*self.shape):
            value = self.objects[i].pdf(x[i])
            if i == ():
                return value
            retval[i] = value
        return retval

    def entropy(self) -> NumpyRealArray:
        retval = np.empty(self.shape, dtype=self.real_dtype)
        for i in np.ndindex(*self.shape):
            retval[i] = self.objects[i].entropy()
        return retval

    def access_object(self, index: tuple[int, ...]) -> T:
        return self.objects[index]
