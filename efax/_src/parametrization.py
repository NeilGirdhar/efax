from __future__ import annotations

from functools import partial, reduce
from itertools import count
from typing import TYPE_CHECKING, Any, Dict, Iterable, Tuple, Type, TypeVar

import jax.numpy as jnp
from tjax import ComplexArray, RealArray, Shape, custom_jvp, jit
from tjax.dataclasses import field_names_values_metadata, fields

from .parameter import Support
from .tools import parameters_dot_product

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
                y_dot = parameters_dot_product(q_dot, p)
                return y, y_dot

            method_jvp.defjvp(ln_jvp)

            setattr(cls, name, method_jvp)

    # New methods ----------------------------------------------------------------------------------
    def __getitem__(self: T, key: Any) -> T:
        fixed_parameters = self.fixed_parameters_mapping()
        sliced_parameters = {name: value[key]
                             for name, value, _ in self.parameters_name_value_support()}
        return type(self)(**sliced_parameters, **fixed_parameters)  # type: ignore

    def flattened(self) -> ComplexArray:
        return reduce(partial(jnp.append, axis=-1),
                      (support.flattened(value)
                       for name, value, support in self.parameters_name_value_support()))

    @classmethod
    def unflattened(cls: Type[T], flattened: ComplexArray, **kwargs: Any) -> T:
        # Solve for dimensions.
        def total_elements(dimensions: int) -> int:
            return sum(support.num_elements(dimensions)
                       for _, support in cls.parameters_name_support())

        target = flattened.shape[-1]
        dimensions = 0
        for dimensions in count():
            te = total_elements(dimensions)
            if te == target:
                break
            if te > target:
                raise ValueError

        # Unflatten.
        consumed = 0
        for name, support in cls.parameters_name_support():
            k = support.num_elements(dimensions)
            kwargs[name] = support.unflattened(flattened[..., consumed: consumed + k], dimensions)
            consumed += k

        return cls(**kwargs)  # type: ignore

    def fixed_parameters_mapping(self) -> Dict[str, Any]:
        return {name: value
                for name, value, metadata in field_names_values_metadata(self)
                if metadata['fixed']}

    def parameters_value_support(self) -> Iterable[Tuple[ComplexArray, Support]]:
        """
        Returns: The value and support of each variable parameter.
        """
        for _, value, metadata in field_names_values_metadata(self):
            if metadata['fixed']:
                continue
            support = metadata['support']
            if not isinstance(support, Support):
                raise TypeError
            yield value, support

    def parameters_name_value(self) -> Iterable[Tuple[str, ComplexArray]]:
        """
        Returns: The name and value of each variable parameter.
        """
        for name, value, metadata in field_names_values_metadata(self):
            if metadata['fixed']:
                continue
            yield name, value

    def parameters_name_value_support(self) -> Iterable[Tuple[str, ComplexArray, Support]]:
        """
        Returns: The name, value, and support of each variable parameter.
        """
        for name, value, metadata in field_names_values_metadata(self):
            if metadata['fixed']:
                continue
            support = metadata['support']
            if not isinstance(support, Support):
                raise TypeError
            yield name, value, support

    @classmethod
    def parameters_name_support(cls) -> Iterable[Tuple[str, Support]]:
        """
        Returns: The name and support of each variable parameter.
        """
        for this_field in fields(cls):
            if this_field.metadata['fixed']:
                continue
            support = this_field.metadata['support']
            if not isinstance(support, Support):
                raise TypeError
            yield this_field.name, support

    # Abstract methods -----------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        raise NotImplementedError


if TYPE_CHECKING:
    from .natural_parametrization import NaturalParametrization
