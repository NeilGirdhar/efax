import math

from ipromise import implements
from jax import numpy as jnp
from tjax import ComplexArray, RealArray

from .exponential_family import ExponentialFamily

__all__ = ['ComplexNormal']


class ComplexNormal(ExponentialFamily):

    def __init__(self) -> None:
        super().__init__(num_parameters=3)

    # Implemented methods --------------------------------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: ComplexArray) -> RealArray:
        eta = q[..., 0]
        precision = q[..., 1]
        pseudo_precision = q[..., 2]

        r = -pseudo_precision / precision
        s = 1.0 / ((jnp.conj(r) * r - 1.0) * precision)
        k = pseudo_precision / precision
        l_eta = (0.5 * eta
                 / ((k * jnp.conj(k) - 1.0) * precision))
        mu = (jnp.conj(l_eta)
              - jnp.conj(pseudo_precision / precision) * l_eta)

        det_s = jnp.real(s)
        det_h = jnp.real(-precision)
        return (- jnp.real(jnp.conj(mu) * mu * precision)
                - jnp.real(mu * mu * pseudo_precision)
                + 0.5 * jnp.log(det_s)
                - 0.5 * jnp.log(det_h)
                + math.log(math.pi))

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: ComplexArray) -> ComplexArray:
        eta = q[..., 0]
        precision = q[..., 1]
        pseudo_precision = q[..., 2]

        r = -pseudo_precision / precision
        s = 1.0 / ((jnp.conj(r) * r - 1.0) * precision)
        u = jnp.conj(r * s)
        k = pseudo_precision / precision
        l_eta = (0.5 * eta
                 / ((k * jnp.conj(k) - 1.0) * precision))
        mu = (jnp.conj(l_eta)
              - jnp.conj(pseudo_precision / precision) * l_eta)
        return jnp.stack([mu, s + jnp.conj(mu) * mu, u + jnp.square(mu)], axis=-1)

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: ComplexArray) -> ComplexArray:
        mean = p[..., 0]
        second_moment = p[..., 1]
        pseudo_second_moment = p[..., 2]
        variance = second_moment - jnp.conj(mean) * mean
        pseudo_variance = pseudo_second_moment - jnp.square(mean)

        r = pseudo_variance.conjugate() / variance
        p_c = variance - (r * pseudo_variance).conjugate()
        p_inv_c = 1.0 / p_c
        precision = -p_inv_c.real
        pseudo_precision = r * p_inv_c
        eta = -2.0 * (precision * mean.conjugate()
                      + pseudo_precision * mean)
        return jnp.stack([eta, precision, pseudo_precision], axis=-1)

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: ComplexArray) -> ComplexArray:
        return jnp.stack([x, jnp.conj(x) * x, jnp.square(x)], axis=-1)
