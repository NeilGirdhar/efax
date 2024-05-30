from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from tjax import JaxArray, JaxComplexArray, JaxRealArray, KeyArray, Shape, abs_square
from tjax.dataclasses import dataclass
from typing_extensions import override

from ...expectation_parametrization import ExpectationParametrization
from ...interfaces.samplable import Samplable
from ...mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ...natural_parametrization import NaturalParametrization
from ...parameter import (ComplexField, RealField, ScalarSupport, Support, complex_field,
                          distribution_parameter)
from ...parametrization import SimpleDistribution


@dataclass
class ComplexNormalNP(HasEntropyNP['ComplexNormalEP'],
                      Samplable,
                      NaturalParametrization['ComplexNormalEP', JaxComplexArray],
                      SimpleDistribution):
    """The natural parametrization of the complex normal distribution.

    Args:
        mean_times_precision: E(x) / Var(x).
        negative_precision: -1 / Var(x).
        pseudo_precision: 1 / PVar(x).

    Where the pseudo-variance is
        PVar(x) = E(x^2 - E(x)^2)

    We know that |PVar(x)| < Var(x) where |x| is the modulus.
    """
    mean_times_precision: JaxComplexArray = distribution_parameter(
        ScalarSupport(ring=complex_field))
    negative_precision: JaxRealArray = distribution_parameter(
            ScalarSupport(ring=RealField(maximum=0.0)))
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
        det_h = -self.negative_precision
        return (-abs_square(mu) * self.negative_precision
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
        r = -self.pseudo_precision / self.negative_precision
        s = 1.0 / ((abs_square(r) - 1.0) * self.negative_precision)
        k = self.pseudo_precision / self.negative_precision
        l_eta = 0.5 * self.mean_times_precision / ((abs_square(k) - 1.0) * self.negative_precision)
        mu = (l_eta.conjugate()
              - (self.pseudo_precision / self.negative_precision).conjugate() * l_eta)
        return r, s, mu

    @override
    @classmethod
    def adjust_support(cls, name: str, **kwargs: JaxArray) -> Support:
        if name != 'pseudo_precision':
            return super().adjust_support(name, **kwargs)
        precision = -kwargs['negative_precision']
        return ScalarSupport(ring=ComplexField(maximum_modulus=jnp.abs(precision)))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)


@dataclass
class ComplexNormalEP(HasEntropyEP[ComplexNormalNP],
                      Samplable,
                      ExpectationParametrization[ComplexNormalNP],
                      SimpleDistribution):
    """The expectation parametrization of the complex normal distribution.

    Args:
        mean: E(x).
        second_moment: E(x * conjugate(x)).
        pseudo_second_moment: E(x^2).
    """
    mean: JaxComplexArray = distribution_parameter(ScalarSupport(ring=complex_field))
    second_moment: JaxRealArray = distribution_parameter(ScalarSupport(ring=RealField(minimum=0.0)))
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

    @override
    @classmethod
    def adjust_support(cls, name: str, **kwargs: JaxArray) -> Support:
        if name != 'pseudo_precision':
            return super().adjust_support(name, **kwargs)
        precision = -kwargs['negative_precision']
        return ScalarSupport(ring=ComplexField(maximum_modulus=jnp.abs(precision)))

    def _multivariate_normal_cov(self) -> JaxRealArray:
        variance = self.second_moment - abs_square(self.mean)
        pseudo_variance = self.pseudo_second_moment - jnp.square(self.mean)
        xx = variance + pseudo_variance.real
        xy = pseudo_variance.imag
        yy = variance - pseudo_variance.real
        xx_xy = jnp.stack([xx, xy], axis=-1)
        yx_yy = jnp.stack([xy, yy], axis=-1)
        return 0.5 * jnp.stack([xx_xy, yx_yy], axis=-2)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        else:
            shape = self.shape
        mn_mean = jnp.stack([self.mean.real, self.mean.imag], axis=-1)
        mn_cov = self._multivariate_normal_cov()
        mn_sample = jax.random.multivariate_normal(key, mn_mean, mn_cov, shape)
        return mn_sample[..., 0] + 1j * mn_sample[..., 1]
