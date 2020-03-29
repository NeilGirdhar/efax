from __future__ import annotations

from typing import Sequence, Tuple, Union

from nptyping import Array

from .dtypes import complex_dtype, real_dtype

__all__ = ['Shape',
           'ShapeLike',
           'SliceLike',
           'Tensor',
           'RealTensor',
           'ComplexTensor']


Shape = Tuple[int, ...]
ShapeLike = Union[int, Sequence[int]]
SliceLike = Tuple[Union[int, None, slice], ...]
Tensor = Union[Array[real_dtype], Array[complex_dtype]]
RealTensor = Array[real_dtype]
ComplexTensor = Array[complex_dtype]
