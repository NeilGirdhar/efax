from __future__ import annotations

from typing import TypeGuard

import numpy as np
from tjax import NumpyComplexArray, NumpyIntegralArray, NumpyRealArray

type NumpyNumericArray = NumpyRealArray | NumpyComplexArray | NumpyIntegralArray


def is_integral(x: NumpyNumericArray) -> TypeGuard[NumpyIntegralArray]:
    return np.isdtype(x.dtype, "integral")


def is_real(x: NumpyNumericArray) -> TypeGuard[NumpyRealArray]:
    return np.isdtype(x.dtype, "real floating")


def is_complex(x: NumpyNumericArray) -> TypeGuard[NumpyComplexArray]:
    return np.isdtype(x.dtype, "complex floating")
