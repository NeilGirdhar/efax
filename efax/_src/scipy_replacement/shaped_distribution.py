from __future__ import annotations

from typing import Any, TypeVar, cast, override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from tjax import NumpyComplexArray, NumpyIntegralArray, NumpyRealArray, Shape

from .base import ScipyDiscreteDistribution, ScipyDistribution

T = TypeVar('T', bound=ScipyDiscreteDistribution | ScipyDistribution)


class ShapedDistribution[T: ScipyDiscreteDistribution | ScipyDistribution]:
    """Allow a distributions with shape."""
    @override
    def __init__(self,
                 shape: Shape,
                 rvs_shape: Shape,
                 rvs_dtype: np.dtype[Any],
                 objects: npt.NDArray[np.object_]
                 ) -> None:
        super().__init__()
        self.shape = shape
        self.rvs_shape = rvs_shape
        self.rvs_dtype = rvs_dtype
        self.real_dtype: np.dtype[Any] = np.real(np.zeros(0, dtype=rvs_dtype)).dtype
        self.objects = objects

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def rvs(self,
            size: int | Shape | None = None,
            random_state: Generator | None = None) -> NumpyRealArray:
        if size is None:
            size = ()
        elif isinstance(size, int):
            size = (size,)
        retval = np.empty(self.shape + size + self.rvs_shape,
                          dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            this_object = cast('T', self.objects[i])
            retval[i] = this_object.rvs(size=size, random_state=random_state)
        return retval

    def pdf(self, x: NumpyComplexArray) -> NumpyRealArray:
        assert x.shape[:self.ndim] == self.shape
        retval = np.empty(x.shape, dtype=self.real_dtype)
        for i in np.ndindex(*self.shape):
            this_object = cast('T', self.objects[i])
            if not isinstance(this_object, ScipyDistribution):
                raise NotImplementedError
            for j in np.ndindex(*x.shape[self.ndim:]):
                value = this_object.pdf(x[*i, *j])
                retval[*i, *j] = value
        return retval

    def pmf(self, x: NumpyIntegralArray) -> NumpyRealArray:
        assert x.shape[:self.ndim] == self.shape
        retval = np.empty(x.shape, dtype=self.real_dtype)
        for i in np.ndindex(*self.shape):
            this_object = cast('T', self.objects[i])
            if not isinstance(this_object, ScipyDiscreteDistribution):
                raise NotImplementedError
            for j in np.ndindex(*x.shape[self.ndim:]):
                value = this_object.pmf(x[*i, *j])
                retval[*i, *j] = value
        return retval

    def entropy(self) -> NumpyRealArray:
        retval = np.empty(self.shape, dtype=self.real_dtype)
        for i in np.ndindex(*self.shape):
            this_object = cast('T', self.objects[i])
            retval[i] = this_object.entropy()
        return retval

    def access_object(self, index: tuple[int, ...]) -> T:
        return self.objects[index]
