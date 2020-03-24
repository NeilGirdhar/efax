"""
This package provides abstract interfaces:
* tensor manipulation (TensorManipulator),
* interval arithmetic (Extrema), and
* slicing and broadcasting (Matching).

The interface is implemented using cpu and gpu routines.
"""
from .annotations import *
from .dtypes import *
from .jax_generator import *
from .tensor_like import *
from .tools import *
