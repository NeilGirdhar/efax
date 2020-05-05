from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Hashable, List, Sequence, Tuple, Union

import numpy as np
from jax import numpy as jnp

__all__ = ['Shape',
           'ShapeLike',
           'SliceLike',
           'Tensor',
           'RealTensor',
           'ComplexTensor',
           'PyTree']


Shape = Tuple[int, ...]
ShapeLike = Union[int, Sequence[int]]
SliceLike = Tuple[Union[int, None, slice], ...]
Tensor = Union[np.ndarray, jnp.ndarray]
RealTensor = Tensor
ComplexTensor = Tensor
PyTree = Union[Tensor,
               'PyTreeLike',
               Tuple['PyTree', ...],
               List['PyTree'],
               Dict[Hashable, 'PyTree'],
               None]


if TYPE_CHECKING:
    from .pytree_like import PyTreeLike
