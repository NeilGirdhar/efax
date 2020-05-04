from __future__ import annotations

from typing import Any, Hashable, Tuple, Type, TypeVar

from jax.tree_util import register_pytree_node

from .annotations import PyTree

__all__ = ['PyTreeLike']


T = TypeVar('T', bound='PyTreeLike')


class PyTreeLike:

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)

        register_pytree_node(cls,
                             cls.tree_flatten,
                             cls.tree_unflatten)

    @classmethod
    def tree_unflatten(cls: Type[T],
                       aux_data: Hashable,
                       values: PyTree,
                       **init_kwargs: Any) -> T:
        """
        Args:
            aux_data:
                Data that will be treated as constant through JAX operations.
            values:
                A JAX PyTree of values from which the object is constructed.
        Returns:
            A constructed object.
        """
        if aux_data is not None:
            raise RuntimeError
        if values is not None:
            raise RuntimeError
        return cls(**init_kwargs)

    def tree_flatten(self) -> Tuple[PyTree, Hashable]:
        """
        Returns:
            values: A JAX PyTree of values representing the object.
            aux_data:
                Data that will be treated as constant through JAX operations.
        """
        # https://github.com/python/mypy/issues/8768
        return None, None  # type: ignore
