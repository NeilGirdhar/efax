from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Tuple, Type, TypeVar

import numpy as np
from chex import Array
from jax import numpy as jnp
from jax.tree_util import tree_map, tree_reduce
from tjax import RealArray, Shape, custom_jvp, jit

from .parameter import parameter_names_axes
from .tools import tree_dot_final

__all__ = ['Parametrization']


T = TypeVar('T', bound='Parametrization')


class Parametrization:
    # Magic methods --------------------------------------------------------------------------------
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        # Apply jit.
        for name in ['log_normalizer',
                     'to_exp',
                     'carrier_measure',
                     'sufficient_statistics',
                     'cross_entropy',
                     'expected_carrier_measure']:
            super_cls = super(cls, cls)
            if not hasattr(cls, name):
                continue
            original_method = getattr(cls, name)
            if hasattr(super_cls, name) and getattr(super_cls, name) is original_method:
                continue  # We only need to jit new methods.
            method = jit(original_method)
            setattr(cls, f'_original_{name}', method)

            if name != 'log_normalizer':
                setattr(cls, name, method)
                continue

            method_jvp: Any = custom_jvp(method)

            def ln_jvp(primals: Tuple[NaturalParametrization[Any]],
                       tangents: Tuple[NaturalParametrization[Any]]) -> Tuple[RealArray, RealArray]:
                q, = primals
                q_dot, = tangents
                y = q.log_normalizer()
                p = q.to_exp()
                y_dot = tree_dot_final(q_dot, p)
                return y, y_dot

            method_jvp.defjvp(ln_jvp)

            setattr(cls, name, method_jvp)

    # New methods ----------------------------------------------------------------------------------
    def flattened(self) -> Array:
        def flatten_parameter(x: Array) -> Array:
            return jnp.reshape(x, (*self.shape(), -1))
        return tree_reduce(partial(jnp.append, axis=-1), tree_map(flatten_parameter, self))

    @classmethod
    def unflattened(cls: Type[T], flattened: Array, **kwargs: Any) -> T:
        # Count the fields with 0, 1, and 2 axes.  Subtract the shape of flattened from the 0 count.
        totals = np.zeros(3, dtype=np.int_)
        totals[0] -= flattened.shape[-1]
        for _, n_axes in parameter_names_axes(cls):
            if not 0 <= n_axes <= 2:
                raise ValueError
            totals[n_axes] += 1

        # Solve the quadratic equation and select the largest positive root.
        roots = np.roots(list(reversed(list(totals))))
        roots = list(roots)
        if not roots:
            root = 1
        else:
            root = int(max(roots))
        if root < 0:
            raise ValueError

        # Unflatten.
        shape = flattened.shape[:-1]
        consumed = 0
        for name, n_axes in parameter_names_axes(cls):
            k = root ** n_axes
            kwargs[name] = np.reshape(flattened[..., consumed: consumed + k],
                                      shape + (root,) * n_axes)
            consumed += k

        return cls(**kwargs)  # type: ignore

    # Abstract methods -----------------------------------------------------------------------------
    def shape(self) -> Shape:
        raise NotImplementedError


if TYPE_CHECKING:
    from .natural_parametrization import NaturalParametrization
