from __future__ import annotations

import math

import jax.random as jr
from array_api_compat import array_namespace
from tjax import JaxArray, JaxComplexArray, JaxRealArray, KeyArray, Shape, abs_square
from tjax.dataclasses import dataclass
from typing_extensions import override

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
        xp = self.array_namespace()
        _, s, mu = self._r_s_mu()
        det_s = s
        det_h = -self.negative_precision
        return (-abs_square(mu) * self.negative_precision
                - xp.real(xp.square(mu) * self.pseudo_precision)
                + 0.5 * xp.log(det_s)
                - 0.5 * xp.log(det_h)
                + math.log(math.pi))

    @override
    def to_exp(self) -> ComplexNormalEP:
        xp = self.array_namespace()
        r, s, mu = self._r_s_mu()
        u = xp.conj(r * s)
        return ComplexNormalEP(mu, s + abs_square(mu), u + xp.square(mu))

    @override
    def carrier_measure(self, x: JaxComplexArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return xp.zeros(x.shape)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxComplexArray, **fixed_parameters: JaxArray
                              ) -> ComplexNormalEP:
        xp = array_namespace(x)
        return ComplexNormalEP(x, abs_square(x), xp.square(x))

    def _r_s_mu(self) -> tuple[JaxComplexArray, JaxRealArray, JaxComplexArray]:
        xp = self.array_namespace()
        r = -self.pseudo_precision / self.negative_precision
        s = xp.reciprocal((abs_square(r) - 1.0) * self.negative_precision)
        k = self.pseudo_precision / self.negative_precision
        l_eta = 0.5 * self.mean_times_precision / ((abs_square(k) - 1.0) * self.negative_precision)
        mu = xp.conj(l_eta) - xp.conj(self.pseudo_precision / self.negative_precision) * l_eta
        return r, s, mu

    @override
    @classmethod
    def adjust_support(cls, name: str, **kwargs: JaxArray) -> Support:
        if name != 'pseudo_precision':
            return super().adjust_support(name, **kwargs)
        precision = -kwargs['negative_precision']
        xp = array_namespace(precision)
        return ScalarSupport(ring=ComplexField(maximum_modulus=xp.abs(precision)))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)


@dataclass
class ComplexNormalEP(HasEntropyEP[ComplexNormalNP],
                      Samplable,
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
        xp = self.array_namespace()
        variance = self.second_moment - abs_square(self.mean)
        pseudo_variance = self.pseudo_second_moment - xp.square(self.mean)

        r = xp.conj(pseudo_variance) / variance
        p_c = variance - xp.conj(r * pseudo_variance)
        p_inv_c = xp.reciprocal(p_c)
        precision = xp.real(-p_inv_c)
        pseudo_precision = r * p_inv_c
        mean_times_precision = -2.0 * (precision * xp.conj(self.mean)
                                       + pseudo_precision * self.mean)
        return ComplexNormalNP(mean_times_precision, precision, pseudo_precision)

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.zeros(self.shape)

    @override
    @classmethod
    def adjust_support(cls, name: str, **kwargs: JaxArray) -> Support:
        if name != 'pseudo_precision':
            return super().adjust_support(name, **kwargs)
        precision = -kwargs['negative_precision']
        xp = array_namespace(precision)
        return ScalarSupport(ring=ComplexField(maximum_modulus=xp.abs(precision)))

    def _multivariate_normal_cov(self) -> JaxRealArray:
        xp = self.array_namespace()
        variance = self.second_moment - abs_square(self.mean)
        pseudo_variance = self.pseudo_second_moment - xp.square(self.mean)
        xx = variance + xp.real(pseudo_variance)
        xy = xp.imag(pseudo_variance)
        yy = variance - xp.real(pseudo_variance)
        xx_xy = xp.stack([xx, xy], axis=-1)
        yx_yy = xp.stack([xy, yy], axis=-1)
        return 0.5 * xp.stack([xx_xy, yx_yy], axis=-2)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        shape = self.shape if shape is None else shape + self.shape
        xp = self.array_namespace()
        mn_mean = xp.stack([xp.real(self.mean), xp.imag(self.mean)], axis=-1)
        mn_cov = self._multivariate_normal_cov()
        mn_sample = jr.multivariate_normal(key, mn_mean, mn_cov, shape)
        return mn_sample[..., 0] + 1j * mn_sample[..., 1]
