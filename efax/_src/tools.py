from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any, Iterable, List

import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from tjax import ComplexArray, ComplexNumeric, RealArray, RealNumeric

__all__: List[str] = []


def np_abs_square(x: ComplexNumeric) -> RealNumeric:
    return np.square(x.real) + np.square(x.imag)  # pyright: ignore


def parameters_dot_product(x: NaturalParametrization[Any, Any], y: Any) -> RealArray:
    def dotted_fields() -> Iterable[RealArray]:
        for (x_value, x_support), (y_value, y_support) in zip(x.parameters_value_support(),
                                                              y.parameters_value_support()):
            axes = x_support.axes()
            assert y_support.axes() == axes
            yield _parameter_dot_product(x_value, y_value, axes)
    return reduce(jnp.add, dotted_fields())


def inverse_softplus(y: RealArray) -> RealArray:
    return jnp.where(y > 80.0,
                     y,
                     jnp.log(jnp.expm1(y)))


ive = tfp.math.bessel_ive
log_ive = tfp.math.log_bessel_ive


def iv(v: RealNumeric, z: RealNumeric) -> RealNumeric:
    return tfp.math.bessel_ive(v, z) / jnp.exp(-jnp.abs(z))


def vectorized_tril(m: RealArray, k: int = 0) -> RealArray:
    n, m_ = m.shape[-2:]
    indices = (..., *np.tril_indices(n, k, m_))
    values = m[indices]
    retval = np.zeros_like(m)
    retval[indices] = values
    return retval


def vectorized_triu(m: RealArray, k: int = 0) -> RealArray:
    n, m_ = m.shape[-2:]
    indices = (..., *np.triu_indices(n, k, m_))
    values = m[indices]
    retval = np.zeros_like(m)
    retval[indices] = values
    return retval


def create_diagonal(m: RealArray) -> RealArray:
    """
    Args:
        m: Has shape (*k, n)
    Returns: Array with shape (*k, n, n) and the elements of m on the diagonals.
    """
    indices = (..., *np.diag_indices(m.shape[-1]))
    retval = np.zeros(m.shape + (m.shape[-1],), dtype=m.dtype)
    retval[indices] = m
    return retval


# Private functions --------------------------------------------------------------------------------
def _parameter_dot_product(x: ComplexArray, y: ComplexArray, n_axes: int) -> RealArray:
    """
    Returns the real component of the dot product of the final n_axes axes of two arrays.
    """
    axes = tuple(range(-n_axes, 0))
    return jnp.sum(x * y, axis=axes).real


if TYPE_CHECKING:
    from .natural_parametrization import NaturalParametrization
