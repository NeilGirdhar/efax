from abc import abstractmethod

from jax.tree_util import register_pytree_node

__all__ = ['TensorLike']


class TensorLike:

    def __init_subclass__(cls):
        def special_unflatten(aux_data, children):
            return cls.from_tensor(children, aux_data)

        register_pytree_node(cls,
                             cls.to_tensor,
                             special_unflatten)

    @classmethod
    @abstractmethod
    def from_tensor(cls, values, aux_data):
        """
        Args:
            values: A JAX pytree of values from which the objects are
                constructed.
            aux_data: Data that will be treated as constant through JAX
                operations.
        Returns:
            A constructed object.
        """
        raise NotImplementedError

    @abstractmethod
    def to_tensor(self):
        """
        Returns:
            values: A JAX pytree of values representing the object.
                aux_data.
            aux_data: Data that will be treated as constant through JAX
                operations.
        """
        raise NotImplementedError
