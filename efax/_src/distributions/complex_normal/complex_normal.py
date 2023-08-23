from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
from tjax import JaxComplexArray, JaxRealArray, Shape, abs_square
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...expectation_parametrization import ExpectationParametrization
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import ScalarSupport, complex_field, distribution_parameter

__all__ = ['ComplexNormalNP', 'ComplexNormalEP']


@dataclass
class ComplexNormalNP(HasEntropyNP['ComplexNormalEP'],
                      NaturalParametrization['ComplexNormalEP', JaxComplexArray]):
    """The natural parametrization of the complex normal distribution.

    Args:
        mean_times_precision: E(x) / Var(x).
        precision: 1 / Var(x).
        pseudo_precision: 1 / E(x^2 - E(x)^2).
    """
    mean_times_precision: JaxComplexArray = distribution_parameter(
        ScalarSupport(ring=complex_field))
    precision: JaxRealArray = distribution_parameter(ScalarSupport())
    pseudo_precision: JaxComplexArray = distribution_parameter(ScalarSupport(ring=complex_field))

    @property
    @override
    def shape(self) -> Shape:
        return self.mean_times_precision.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=complex_field)

    @override
    def log_normalizer(self) -> JaxRealArray:
        _, s, mu = self._r_s_mu()
        det_s = s
        det_h = -self.precision
        return (-abs_square(mu) * self.precision
                - (jnp.square(mu) * self.pseudo_precision).real
                + 0.5 * jnp.log(det_s)
                - 0.5 * jnp.log(det_h)
                + math.log(math.pi))

    @override
    def to_exp(self) -> ComplexNormalEP:
        r, s, mu = self._r_s_mu()
        u = (r * s).conjugate()
        return ComplexNormalEP(mu, s + abs_square(mu), u + jnp.square(mu))

    @override
    def carrier_measure(self, x: JaxComplexArray) -> JaxRealArray:
        return jnp.zeros(x.shape)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxComplexArray, **fixed_parameters: Any
                              ) -> ComplexNormalEP:
        return ComplexNormalEP(x, abs_square(x), jnp.square(x))

    def _r_s_mu(self) -> tuple[JaxComplexArray, JaxRealArray, JaxComplexArray]:
        r = -self.pseudo_precision / self.precision
        s = 1.0 / ((abs_square(r) - 1.0) * self.precision)
        k = self.pseudo_precision / self.precision
        l_eta = 0.5 * self.mean_times_precision / ((abs_square(k) - 1.0) * self.precision)
        mu = l_eta.conjugate() - (self.pseudo_precision / self.precision).conjugate() * l_eta
        return r, s, mu


@dataclass
class ComplexNormalEP(HasEntropyEP[ComplexNormalNP],
                      ExpectationParametrization[ComplexNormalNP]):
    """The expectation parametrization of the complex normal distribution.

    Args:
        mean: E(x).
        second_moment: E(x * conjugate(x)).
        pseudo_second_moment: E(x^2).
    """
    mean: JaxComplexArray = distribution_parameter(ScalarSupport(ring=complex_field))
    second_moment: JaxRealArray = distribution_parameter(ScalarSupport())
    pseudo_second_moment: JaxComplexArray = distribution_parameter(
        ScalarSupport(ring=complex_field))

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=complex_field)

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[ComplexNormalNP]:
        return ComplexNormalNP

    @override
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

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.shape)
