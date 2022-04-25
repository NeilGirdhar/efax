from __future__ import annotations

import math
from typing import Tuple, Type

import jax.numpy as jnp
from tjax import ComplexArray, RealArray, Shape, abs_square
from tjax.dataclasses import dataclass

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter

__all__ = ['ComplexNormalNP', 'ComplexNormalEP']


@dataclass
class ComplexNormalNP(NaturalParametrization['ComplexNormalEP', ComplexArray]):
    mean_times_precision: ComplexArray = distribution_parameter(ScalarSupport(is_complex=True))
    precision: RealArray = distribution_parameter(ScalarSupport())
    pseudo_precision: ComplexArray = distribution_parameter(ScalarSupport(is_complex=True))

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean_times_precision.shape

    def log_normalizer(self) -> RealArray:
        _, s, mu = self._r_s_mu()
        det_s = s
        det_h = -self.precision
        return (-abs_square(mu) * self.precision
                - (jnp.square(mu) * self.pseudo_precision).real
                + 0.5 * jnp.log(det_s)
                - 0.5 * jnp.log(det_h)
                + math.log(math.pi))

    def to_exp(self) -> ComplexNormalEP:
        r, s, mu = self._r_s_mu()
        u = (r * s).conjugate()
        return ComplexNormalEP(mu, s + abs_square(mu), u + jnp.square(mu))

    def carrier_measure(self, x: ComplexArray) -> RealArray:
        return jnp.zeros(x.shape)

    def sufficient_statistics(self, x: ComplexArray) -> ComplexNormalEP:
        return ComplexNormalEP(x, abs_square(x), jnp.square(x))

    def _r_s_mu(self) -> Tuple[ComplexArray, RealArray, ComplexArray]:
        r = -self.pseudo_precision / self.precision
        s = 1.0 / ((abs_square(r) - 1.0) * self.precision)
        k = self.pseudo_precision / self.precision
        l_eta = 0.5 * self.mean_times_precision / ((abs_square(k) - 1.0) * self.precision)
        mu = l_eta.conjugate() - (self.pseudo_precision / self.precision).conjugate() * l_eta
        return r, s, mu


@dataclass
class ComplexNormalEP(ExpectationParametrization[ComplexNormalNP]):
    mean: ComplexArray = distribution_parameter(ScalarSupport(is_complex=True))
    second_moment: RealArray = distribution_parameter(ScalarSupport())
    pseudo_second_moment: ComplexArray = distribution_parameter(ScalarSupport(is_complex=True))

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean.shape

    @classmethod
    def natural_parametrization_cls(cls) -> Type[ComplexNormalNP]:
        return ComplexNormalNP

    def to_nat(self) -> ComplexNormalNP:
        variance = self.second_moment - abs_square(self.mean)
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
        return jnp.zeros(self.shape)
