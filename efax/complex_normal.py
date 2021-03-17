from __future__ import annotations

import math

import jax.numpy as jnp
from tjax import Array, ComplexArray, RealArray, Shape, dataclass

from .expectation_parametrization import ExpectationParametrization
from .natural_parametrization import NaturalParametrization
from .parameter import ScalarSupport, distribution_parameter

__all__ = ['ComplexNormalNP', 'ComplexNormalEP']


@dataclass
class ComplexNormalNP(NaturalParametrization['ComplexNormalEP']):
    mean_times_precision: ComplexArray = distribution_parameter(ScalarSupport())
    precision: ComplexArray = distribution_parameter(ScalarSupport())
    pseudo_precision: ComplexArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean_times_precision.shape

    def log_normalizer(self) -> RealArray:
        r = -self.pseudo_precision / self.precision
        s = 1.0 / ((jnp.conj(r) * r - 1.0) * self.precision)
        k = self.pseudo_precision / self.precision
        l_eta = (0.5 * self.mean_times_precision
                 / ((k * jnp.conj(k) - 1.0) * self.precision))
        mu = (jnp.conj(l_eta)
              - jnp.conj(self.pseudo_precision / self.precision) * l_eta)

        det_s = jnp.real(s)
        det_h = jnp.real(-self.precision)
        return (- jnp.real(jnp.conj(mu) * mu * self.precision)
                - jnp.real(mu * mu * self.pseudo_precision)
                + 0.5 * jnp.log(det_s)
                - 0.5 * jnp.log(det_h)
                + math.log(math.pi))

    def to_exp(self) -> ComplexNormalEP:
        r = -self.pseudo_precision / self.precision
        s = 1.0 / ((jnp.conj(r) * r - 1.0) * self.precision)
        u = jnp.conj(r * s)
        k = self.pseudo_precision / self.precision
        l_eta = (0.5 * self.mean_times_precision
                 / ((k * jnp.conj(k) - 1.0) * self.precision))
        mu = (jnp.conj(l_eta)
              - jnp.conj(self.pseudo_precision / self.precision) * l_eta)
        return ComplexNormalEP(mu,
                               s + jnp.conj(mu) * mu,
                               u + jnp.square(mu))

    def carrier_measure(self, x: Array) -> RealArray:
        return jnp.zeros(x.shape)

    def sufficient_statistics(self, x: Array) -> ComplexNormalEP:
        return ComplexNormalEP(x, jnp.conj(x) * x, jnp.square(x))


@dataclass
class ComplexNormalEP(ExpectationParametrization[ComplexNormalNP]):
    mean: ComplexArray = distribution_parameter(ScalarSupport())
    second_moment: RealArray = distribution_parameter(ScalarSupport())
    pseudo_second_moment: ComplexArray = distribution_parameter(ScalarSupport())

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean.shape

    def to_nat(self) -> ComplexNormalNP:
        variance = self.second_moment - jnp.conj(self.mean) * self.mean
        pseudo_variance = self.pseudo_second_moment - jnp.square(self.mean)

        r = pseudo_variance.conjugate() / variance
        p_c = variance - (r * pseudo_variance).conjugate()
        p_inv_c = 1.0 / p_c
        precision = -p_inv_c.real
        pseudo_precision = r * p_inv_c
        mean_times_precision = -2.0 * (precision * self.mean.conjugate()
                                       + pseudo_precision * self.mean)
        return ComplexNormalNP(mean_times_precision, precision, pseudo_precision)

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape())
