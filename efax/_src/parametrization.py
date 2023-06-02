from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import fields
from functools import partial, reduce
from itertools import count
from typing import Any

import jax.numpy as jnp
from tjax import JaxAbstractClass, JaxComplexArray, JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import Self

from .parameter import Support

__all__ = ['Parametrization']


@dataclass
class Parametrization(JaxAbstractClass):
    def __getitem__(self, key: Any) -> Self:
        parameters = {field.name: getattr(self, field.name)[key]
                      for field in fields(self)}
        return type(self)(**parameters)

    def flattened(self) -> JaxRealArray:
        return reduce(partial(jnp.append, axis=-1),
                      (support.flattened(value)
                       for _, value, support in self.parameters_name_value_support()))

    @classmethod
    def unflattened(cls, flattened: JaxRealArray, **fixed_parameters: Any) -> Self:
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
        kwargs = fixed_parameters
        consumed = 0
        for name, support in cls.parameters_name_support():
            k = support.num_elements(dimensions)
            kwargs[name] = support.unflattened(flattened[..., consumed: consumed + k], dimensions)
            consumed += k

        return cls(**kwargs)

    def fixed_parameters(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name)
            for field in fields(self)
            if field.metadata['fixed']}

    def parameters_value_support(self) -> Iterable[tuple[JaxComplexArray, Support]]:
        """The value and support of each variable parameter."""
        for field in fields(self):
            value = getattr(self, field.name)
            metadata = field.metadata
            if metadata['fixed']:
                continue
            support = metadata['support']
            if not isinstance(support, Support):
                raise TypeError
            yield value, support

    def parameters_name_value(self) -> Iterable[tuple[str, JaxComplexArray]]:
        """The name and value of each variable parameter."""
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            metadata = field.metadata
            if metadata['fixed']:
                continue
            yield name, value

    def parameters_name_value_support(self) -> Iterable[tuple[str, JaxComplexArray, Support]]:
        """The name, value, and support of each variable parameter."""
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            metadata = field.metadata
            if metadata['fixed']:
                continue
            support = metadata['support']
            if not isinstance(support, Support):
                raise TypeError
            yield name, value, support

    @classmethod
    def parameters_name_support(cls) -> Iterable[tuple[str, Support]]:
        """The name and support of each variable parameter."""
        for field in fields(cls):
            metadata = field.metadata
            if metadata['fixed']:
                continue
            support = metadata['support']
            if not isinstance(support, Support):
                raise TypeError
            yield field.name, support

    @property
    @abstractmethod
    def shape(self) -> Shape:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def domain_support(cls) -> Support:
        raise NotImplementedError
