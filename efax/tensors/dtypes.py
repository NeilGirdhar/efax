from jax import numpy as jnp
from jax.dtypes import canonicalize_dtype

__all__ = ['real_dtype',
           'complex_dtype',
           'default_rtol',
           'default_atol',
           'default_tols']


# This may one day not require canonicalize_dtype.  Check if jnp.float_ is
# still np.float64.  One of the downsides of calling canonicalize_dtype here is
# that this file must be imported after config.update("jax_enable_x64", True)
# or else canonicalize_dtype will already have cached the default float width.
real_dtype = canonicalize_dtype(jnp.float_).type
complex_dtype = canonicalize_dtype(jnp.complex_).type


default_rtol = {jnp.float32: 1e-4,
                jnp.float64: 1e-5}[real_dtype]
default_atol = {jnp.float32: 1e-6,
                jnp.float64: 1e-8}[real_dtype]
default_tols = dict(rtol=default_rtol, atol=default_atol)
