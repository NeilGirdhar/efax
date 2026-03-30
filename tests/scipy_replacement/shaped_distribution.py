from __future__ import annotations

from typing import Any, TypeVar, cast, override

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from tjax import NumpyComplexArray, NumpyIntegralArray, NumpyRealArray, Shape

from .base import ScipyDiscreteDistribution, ScipyDistribution

T = TypeVar("T", bound=ScipyDiscreteDistribution | ScipyDistribution)


class ShapedDistribution[T: ScipyDiscreteDistribution | ScipyDistribution]:
    """Allow a distributions with shape."""

    @override
    def __init__(
        self,
        shape: Shape,
        rvs_shape: Shape,
        rvs_dtype: np.dtype[Any],
        objects: npt.NDArray[np.object_],
        *,
        multivariate: bool,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.rvs_shape = rvs_shape
        self.rvs_dtype = rvs_dtype
        self.real_dtype: np.dtype[Any] = np.real(np.zeros(0, dtype=rvs_dtype)).dtype
        self.objects = objects
        self.multivariate = multivariate

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def sample(self, shape: Shape = (), *, rng: Generator | None = None) -> NumpyRealArray:
        retval = np.empty(self.shape + shape + self.rvs_shape, dtype=self.rvs_dtype)
        for i in np.ndindex(*self.shape):
            this_object = cast("T", self.objects[i])
            retval[i] = (
                this_object.sample(shape=shape, rng=rng)
                if hasattr(this_object, "sample")
                else this_object.rvs(size=shape, random_state=rng)
            )
        return retval

    def pdf(self, x: NumpyComplexArray) -> NumpyRealArray:
        assert x.shape[: self.ndim] == self.shape
        event_ndim = len(self.rvs_shape) if self.multivariate else 0
        final_shape = x.shape[:-event_ndim] if event_ndim else x.shape
        retval = np.empty(final_shape, dtype=self.real_dtype)
        for i in np.ndindex(*self.shape):
            this_object = cast("T", self.objects[i])
            if not hasattr(this_object, "pdf"):
                raise NotImplementedError
            j_range = x.shape[self.ndim : -event_ndim] if event_ndim else x.shape[self.ndim :]
            for j in np.ndindex(*j_range):
                if event_ndim:
                    x_ij = x[*i, *j, ...]
                    assert x_ij.ndim == event_ndim
                else:
                    x_ij = x[*i, *j]
                    assert x_ij.ndim == 0
                value = this_object.pdf(x_ij)
                retval[*i, *j] = value
        return retval

    def pmf(self, x: NumpyIntegralArray) -> NumpyRealArray:
        assert x.shape[: self.ndim] == self.shape
        retval = np.empty(x.shape, dtype=self.real_dtype)
        for i in np.ndindex(*self.shape):
            this_object = cast("T", self.objects[i])
            if not hasattr(this_object, "pmf"):
                raise NotImplementedError
            for j in np.ndindex(*x.shape[self.ndim :]):
                value = this_object.pmf(x[*i, *j])
                retval[*i, *j] = value
        return retval

    def entropy(self) -> NumpyRealArray:
        retval = np.empty(self.shape, dtype=self.real_dtype)
        for i in np.ndindex(*self.shape):
            this_object = cast("T", self.objects[i])
            retval[i] = this_object.entropy()
        return retval

    def access_object(self, index: tuple[int, ...]) -> T:
        return self.objects[index]
