from collections.abc import Sequence
from enum import Enum
from functools import partial
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
from jax import Array, core, grad, jit, lax, vmap
from jax._src import dtypes, xla_bridge
from jax._src.interpreters import batching
from jax._src.lax.lax import _const, _float, _isnan, standard_naryop
from jax._src.lax.special import _any, _up_and_broadcast
from jax._src.random import _check_prng_key, _check_shape, _gamma_batching_rule, _gamma_impl
from jax._src.typing import ArrayLike, DTypeLike
from jax.interpreters import ad, mlir
from jax.lax import (add, bitwise_and, bitwise_not, bitwise_or, cond, digamma, div, eq, exp,
                     full_like, gt, le, lgamma, log, lt, mul, ne, neg, reciprocal, scan, select,
                     sub, while_loop)
from tjax import JaxBooleanArray, JaxRealArray

__all__ = ['random_gamma']


RealArray = ArrayLike
Shape = Sequence[int]
DTypeLikeFloat = DTypeLike
class IgammaMode(Enum):
  VALUE = 1
  DERIVATIVE = 2
  SAMPLE_DERIVATIVE = 3


def _igammac_continued_fraction(ax, x, a, enabled, dtype, mode, hessian):
    eps = dtypes.finfo(dtype).eps

    def cond_fn(vals):
        enabled, _ans, _t, _y, _x, c, *_ = vals
        return bitwise_and(c < _const(c, 2000), _any(enabled))

    def body_fn(vals):
        (enabled, ans, t, y, z, c, pkm1, qkm1, pkm2, qkm2,
            dpkm2_da, dqkm2_da, dpkm1_da, dqkm1_da, dans_da) = vals

        c = c + _const(c, 1)
        y = y + _const(y, 1)
        z = z + _const(z, 2)
        yc = y * c
        pk = pkm1 * z - pkm2 * yc
        qk = qkm1 * z - qkm2 * yc
        qk_is_nonzero = ne(qk, _const(qk, 0))
        r = pk / qk

        t = select(qk_is_nonzero, abs(div(sub(ans, r), r)), full_like(r, 1))
        ans = select(qk_is_nonzero, r, ans)

        dpk_da = dpkm1_da * z - pkm1 - dpkm2_da * yc + pkm2 * c
        dqk_da = dqkm1_da * z - qkm1 - dqkm2_da * yc + qkm2 * c
        dans_da_new = select(qk_is_nonzero, div(dpk_da - ans * dqk_da, qk), dans_da)
        grad_conditional = select(qk_is_nonzero,
                                  abs(dans_da_new - dans_da),
                                  full_like(dans_da, 1))

        pkm2 = pkm1
        pkm1 = pk
        qkm2 = qkm1
        qkm1 = qk

        dpkm2_da = dpkm1_da
        dqkm2_da = dqkm1_da
        dpkm1_da = dpk_da
        dqkm1_da = dqk_da

        rescale = gt(abs(pk), reciprocal(_const(pk, eps)))
        pkm2 = select(rescale, mul(pkm2, _const(pkm2, eps)), pkm2)
        pkm1 = select(rescale, mul(pkm1, _const(pkm1, eps)), pkm1)
        qkm2 = select(rescale, mul(qkm2, _const(qkm2, eps)), qkm2)
        qkm1 = select(rescale, mul(qkm1, _const(qkm1, eps)), qkm1)

        dpkm2_da = select(rescale, mul(dpkm2_da, _const(dpkm2_da, eps)), dpkm2_da)
        dqkm2_da = select(rescale, mul(dqkm2_da, _const(dqkm2_da, eps)), dqkm2_da)
        dpkm1_da = select(rescale, mul(dpkm1_da, _const(dpkm1_da, eps)), dpkm1_da)
        dqkm1_da = select(rescale, mul(dqkm1_da, _const(dqkm1_da, eps)), dqkm1_da)

        if mode == IgammaMode.VALUE:
            conditional = bitwise_and(enabled, t > eps)
        else:
            conditional = bitwise_and(enabled,
                                      grad_conditional > _const(grad_conditional, eps))

        return (conditional,
                select(enabled, ans, vals[1]),
                select(enabled, t, vals[2]),
                select(enabled, y, vals[3]),
                select(enabled, z, vals[4]),
                c,
                select(enabled, pkm1, vals[6]),
                select(enabled, qkm1, vals[7]),
                select(enabled, pkm2, vals[8]),
                select(enabled, qkm2, vals[9]),
                select(enabled, dpkm2_da, vals[10]),
                select(enabled, dqkm2_da, vals[11]),
                select(enabled, dpkm1_da, vals[12]),
                select(enabled, dqkm1_da, vals[13]),
                select(enabled, dans_da_new, vals[14]))

    y = _const(a, 1) - a
    z = x + y + _const(x, 1)
    c = _const(x, 0)
    pkm2 = full_like(x, 1)
    qkm2 = x
    pkm1 = x + _const(x, 1)
    qkm1 = z * x
    ans = pkm1 / qkm1
    t = full_like(x, 1)
    dpkm2_da = full_like(x, 0)
    dqkm2_da = full_like(x, 0)
    dpkm1_da = full_like(x, 0)
    dqkm1_da = -x
    dans_da = (dpkm1_da - ans * dqkm1_da) / qkm1
    init_vals = (enabled,    ans,        t,        y,        z,
                 c,        pkm1,     qkm1,     pkm2,     qkm2,
                 dpkm2_da, dqkm2_da, dpkm1_da, dqkm1_da, dans_da)

    vals = (_while_loop_scan(cond_fn, body_fn, init_vals, 256)
            if hessian
            else while_loop(cond_fn, body_fn, init_vals))
    ans = vals[1]
    if mode == IgammaMode.VALUE:
        return ans *    ax
    dans_da = vals[14]
    dlogax_da = log(x) -    digamma(a)

    if mode == IgammaMode.DERIVATIVE:
        return mul(ax, add(mul(ans, dlogax_da), dans_da))
    if mode == IgammaMode.SAMPLE_DERIVATIVE:
        return neg(add(dans_da, mul(ans, dlogax_da)) * x)
    msg = f'Invalid mode: {mode}'
    raise ValueError(msg)


def _igamma_series(ax: JaxRealArray,
                   x: JaxRealArray,
                   a: JaxRealArray,
                   enabled: JaxBooleanArray,
                   dtype: Any,
                   mode: IgammaMode,
                   hessian) -> JaxRealArray:
    def cond_fn(vals):
        return _any(vals[0])

    def body_fn(vals):
        enabled, r, c, ans, x, dc_da, dans_da = vals

        r = r + _const(r, 1.)
        dc_da = dc_da * (x / r) - (c * x) / (r * r)
        dans_da = dans_da + dc_da
        c = c * (x / r)
        ans = ans + c

        if mode == IgammaMode.VALUE:
            conditional = bitwise_and(enabled, c / ans > dtypes.finfo(dtype).eps)
        else:
            conditional = bitwise_and(enabled,
                                      abs(dc_da / dans_da) >    dtypes.finfo(dtype).eps)

        # TODO: Make this a vmap. Might be tricky with the imports.
        return (
            conditional,
            select(enabled, r, vals[1]),
            select(enabled, c, vals[2]),
            select(enabled, ans, vals[3]),
            select(enabled, x, vals[4]),
            select(enabled, dc_da, vals[5]),
            select(enabled, dans_da, vals[6]),
        )

    init_vals = (
        enabled, a, full_like(a, 1), full_like(a, 1), x, full_like(a, 0),
        full_like(a, 0),
    )

    vals = (_while_loop_scan(cond_fn, body_fn, init_vals, 256)
            if hessian
            else while_loop(cond_fn, body_fn, init_vals))
    ans = vals[3]
    dans_da = vals[6]

    if mode == IgammaMode.VALUE:
        return (ans * ax) / a

    dlogax_da = log(x) - digamma(a + _const(a, 1))

    if mode == IgammaMode.DERIVATIVE:
        return ax * (ans * dlogax_da + dans_da) / a
    if mode == IgammaMode.SAMPLE_DERIVATIVE:
        return -(dans_da + ans * dlogax_da) * x / a
    raise ValueError("Invalid IgammaMode")


def _while_loop_scan(cond_fun, body_fun, init_val, max_iter):
    """Scan-based implementation (jit ok, reverse-mode autodiff ok)."""
    def _iter(val):
        next_val = body_fun(val)
        next_cond = cond_fun(next_val)
        return next_val, next_cond

    def _fun(tup, it):
        val, _cond = tup
        # When cond is met, we start doing no-ops.
        return cond(_cond, _iter, lambda x: (x, False), val), it

    init = (init_val, cond_fun(init_val))
    return scan(_fun, init, None, length=max_iter)[0][0]


def random_gamma_grad_impl(a: JaxRealArray,
                           x: JaxRealArray,
                           *,
                           dtype: Any,
                           hessian: bool = False) -> JaxRealArray:
    is_nan = bitwise_or(_isnan(a), _isnan(x))
    x_is_zero = eq(x, full_like(x,0))
    domain_error = bitwise_or(lt(x, full_like(x,0)), le(a, full_like(a,0)))
    use_igammac = bitwise_and(gt(x, full_like(x,1)), gt(x, a))
    ax = a * log(x) - x - lgamma(a)
    underflow = lt(ax, -log(dtypes.finfo(a.dtype).max))
    ax = exp(ax)
    enabled = bitwise_not(bitwise_or(bitwise_or(bitwise_or
    (x_is_zero, domain_error), underflow), is_nan))
    output = select(use_igammac,
                    -_igammac_continued_fraction(ax, x, a, bitwise_and(enabled, use_igammac),
                                                 dtype, IgammaMode.SAMPLE_DERIVATIVE,
                                                 hessian=hessian),
                    _igamma_series(ax, x, a, bitwise_and(enabled, bitwise_not(use_igammac)),
                                   dtype, IgammaMode.SAMPLE_DERIVATIVE, hessian=hessian))
    output = select(x_is_zero, full_like(output,0), output)
    return select(bitwise_or(domain_error, is_nan),
                  full_like(a, float('nan')), output)


def random_gamma_hessian_a(g, a, x, *, dtype):
    return grad(random_gamma_grad_impl, argnums=0)(a, x, dtype=dtype, hessian=True)


def random_gamma_hessian_x(g, a, x, *, dtype):
    return grad(random_gamma_grad_impl, argnums=1)(a, x, dtype=dtype, hessian=True)


random_gamma_grad_p = standard_naryop([_float, _float], 'random_gamma_grad')
mlir.register_lowering(random_gamma_grad_p,
                       mlir.lower_fun(_up_and_broadcast(random_gamma_grad_impl),
                                      multiple_results=False))
ad.defjvp(random_gamma_grad_p,
          _up_and_broadcast(random_gamma_hessian_a),
          _up_and_broadcast(random_gamma_hessian_x))


def random_gamma_grad(a: ArrayLike, x: ArrayLike) -> Array:
  r"""Elementwise derivative of samples from `Gamma(a, 1)`."""
  return random_gamma_grad_p.bind(a, x)


def _gamma_grad(sample, a, *, log_space):
    samples = jnp.reshape(sample, -1)
    alphas = jnp.reshape(a, -1)
    if log_space:
        # d[log(sample)] = d[sample] / sample
        # This requires computing exp(log_sample), which may be zero due to float roundoff.
        # In this case, correct it to smallest representable float.
        samples = lax.exp(samples)
        zero = _const(sample, 0)
        tiny = lax.full_like(samples, jnp.finfo(samples.dtype).tiny)
        samples = lax.select(lax.eq(samples, zero), tiny, samples)
        def gamma_grad(alpha, sample):
          return random_gamma_grad(alpha, sample) / sample
    else:
        gamma_grad = random_gamma_grad
    if xla_bridge.get_backend().platform == 'cpu':
        grads = lax.map(lambda args: gamma_grad(*args), (alphas, samples))
    else:
        grads = vmap(gamma_grad)(alphas, samples)
    return grads.reshape(np.shape(a))


random_gamma_p = core.Primitive('random_gamma')
random_gamma_p.def_impl(_gamma_impl)
random_gamma_p.def_abstract_eval(lambda key, a, **_: core.raise_to_shaped(a))
ad.defjvp2(
    random_gamma_p, None,
    lambda tangent, ans, key, a, **kwds: tangent * _gamma_grad(ans, a, **kwds))
mlir.register_lowering(random_gamma_p, mlir.lower_fun(
    partial(_gamma_impl, use_vmap=True),
    multiple_results=False))
mlir.register_lowering(random_gamma_p, mlir.lower_fun(
    partial(_gamma_impl, use_vmap=False),
    multiple_results=False), platform='cpu')
batching.primitive_batchers[random_gamma_p] = _gamma_batching_rule


@partial(jit, static_argnames=('shape', 'dtype', 'log_space'))
def _gamma(key, a, shape, dtype, log_space=False) -> Array:
    if shape is None:
        shape = np.shape(a)
    else:
        _check_shape("gamma", shape, np.shape(a))

    a = lax.convert_element_type(a, dtype)
    if np.shape(a) != shape:
        a = jnp.broadcast_to(a, shape)
    return random_gamma_p.bind(key, a, log_space=log_space)


def random_gamma(key: Array,
                 a: ArrayLike,
                 shape: Optional[Shape] = None,
                 dtype: DTypeLikeFloat = dtypes.float_) -> Array:
    r"""Sample Gamma random values with given shape and float dtype.

    The values are distributed according the the probability density function:

    .. math::
         f(x;a) \propto x^{a - 1} e^{-x}

    on the domain :math:`0 \le x < \infty`, with :math:`a > 0`.

    This is the standard gamma density, with a unit scale/rate parameter.
    Dividing the sample output by the rate is equivalent to sampling from
    *gamma(a, rate)*, and multiplying the sample output by the scale is equivalent
    to sampling from *gamma(a, scale)*.

    Args:
        key: a PRNG key used as the random key.
        a: a float or array of floats broadcast-compatible with ``shape``
            representing the parameter of the distribution.
        shape: optional, a tuple of nonnegative integers specifying the result
            shape. Must be broadcast-compatible with ``a``. The default (None)
            produces a result shape equal to ``a.shape``.
        dtype: optional, a float dtype for the returned values (default float64 if
            jax_enable_x64 is true, otherwise float32).

    Returns:
        A random array with the specified dtype and with shape given by ``shape`` if
        ``shape`` is not None, or else by ``a.shape``.

    See Also:
        loggamma : sample gamma values in log-space, which can provide improved
            accuracy for small values of ``a``.
    """
    key, _ = _check_prng_key(key)
    if not dtypes.issubdtype(dtype, np.floating):
        msg = f'dtype argument to `gamma` must be a float dtype, got {dtype}'
        raise ValueError(msg)
    dtype = dtypes.canonicalize_dtype(dtype)
    if shape is not None:
        shape = core.canonicalize_shape(shape)
    return _gamma(key, a, shape=shape, dtype=dtype)
