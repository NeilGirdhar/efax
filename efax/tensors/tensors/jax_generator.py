from typing import Any, Optional

import jax.random

from .tensor_like import TensorLike

__all__ = ['Generator']


class Generator(TensorLike):

    def __init__(self,
                 *,
                 seed: Optional[int] = None,
                 key: Optional[Any] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.key = (key
                    if key is not None
                    else jax.random.PRNGKey(seed))

    # New methods -------------------------------------------------------------
    def split_out_subkey(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def normal(self, std_dev, shape=None):
        subkey = self.split_out_subkey()
        return std_dev * jax.random.normal(subkey, shape)

    def __repr__(self):
        return f"{type(self).__name__}({self.key})"

    # Overridden methods ------------------------------------------------------
    @classmethod
    def from_tensor(cls, values, aux_data):
        return cls(key=values)

    def to_tensor(self):
        return self.key, None
