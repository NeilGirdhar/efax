from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

__all__ = ['real_dtype',
           'complex_dtype',
           'demote_dtype']


# The type of values and times (real and complex) and keys.
real_dtype = np.float32

# These constants are calculated from the above constants.
complex_dtype = {np.float32: np.complex64,
                 np.float64: np.complex128}[real_dtype]


def demote_dtype(array: Tensor):
    dtype = array.dtype
    if dtype in [np.float32, np.float64]:
        if dtype == real_dtype:
            return array
        return array.astype(real_dtype)
    if dtype in [np.complex64, np.complex128]:
        if dtype == complex_dtype:
            return array
        return array.astype(complex_dtype)
    raise ValueError


if TYPE_CHECKING:
    from .annotations import Tensor
