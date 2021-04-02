from __future__ import annotations

from functools import partial, reduce
from itertools import count
from typing import TYPE_CHECKING, Any, Dict, Tuple, Type, TypeVar

import jax.numpy as jnp
from chex import Array
from tjax import RealArray, Shape, custom_jvp, jit

from .parameter import (field_names_values_metadata, parameter_names_support,
                        parameter_names_values_support)
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
    def __getitem__(self: T, key: Any) -> T:
        uk = self.unflattened_kwargs()
        sliced_parameters = {name: value[key]
                             for name, value, _ in parameter_names_values_support(self)}
        return type(self)(**sliced_parameters, **uk)  # type: ignore

    def unflattened_kwargs(self) -> Dict[Any, Any]:
        return {name: value
                for name, value, metadata in field_names_values_metadata(self)
                if 'support' not in metadata}

    def flattened(self) -> Array:
        return reduce(partial(jnp.append, axis=-1),
                      (support.flattened(value)
                       for name, value, support in parameter_names_values_support(self)))

    @classmethod
    def unflattened(cls: Type[T], flattened: Array, **kwargs: Any) -> T:
        # Solve for dimensions.
        def total_elements(dimensions: int) -> int:
            return sum(support.num_elements(dimensions)
                       for _, support in parameter_names_support(cls))

        target = flattened.shape[-1]
        for dimensions in count():
            te = total_elements(dimensions)
            if te == target:
                break
            if te > target:
                raise ValueError

        # Unflatten.
        consumed = 0
        for name, support in parameter_names_support(cls):
            k = support.num_elements(dimensions)
            kwargs[name] = support.unflattened(flattened[..., consumed: consumed + k], dimensions)
            consumed += k

        return cls(**kwargs)  # type: ignore

    # Abstract methods -----------------------------------------------------------------------------
    def shape(self) -> Shape:
        raise NotImplementedError


if TYPE_CHECKING:
    from .natural_parametrization import NaturalParametrization
