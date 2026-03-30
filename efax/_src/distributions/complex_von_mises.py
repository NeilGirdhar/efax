from __future__ import annotations

import math
from typing import cast, override

import jax.scipy.special as jss
from array_api_compat import array_namespace
from tjax import JaxArray, JaxComplexArray, JaxRealArray, Shape, inverse_softplus, softplus
from tjax.dataclasses import dataclass

from efax._src.mixins.exp_to_nat.exp_to_nat import ExpToNat
from efax._src.mixins.has_entropy import HasEntropyEP, HasEntropyNP
from efax._src.natural_parametrization import NaturalParametrization
from efax._src.parameter import ComplexField, ScalarSupport, complex_field, distribution_parameter
from efax._src.parametrization import SimpleDistribution


@dataclass
class ComplexVonMisesNP(
    HasEntropyNP["ComplexVonMisesEP"],
    NaturalParametrization["ComplexVonMisesEP", JaxComplexArray],
    SimpleDistribution,
):
    """The natural parametrization of the von Mises distribution.

    Args:
        mean_times_concentration: Complex phasor η = κ exp(i μ).
    """

    mean_times_concentration: JaxComplexArray = distribution_parameter(
        ScalarSupport(ring=complex_field)
    )

    @property
    @override
    def shape(self) -> Shape:
        return self.mean_times_concentration.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=complex_field)

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = array_namespace(self)
        kappa = xp.abs(self.mean_times_concentration)
        return kappa + math.log(2.0 * math.pi) + xp.log(jss.i0e(kappa))

    @override
    def to_exp(self) -> ComplexVonMisesEP:
        xp = array_namespace(self)
        q = self.mean_times_concentration
        kappa = xp.abs(q)
        return ComplexVonMisesEP(xp.where(kappa == 0.0, q, q * (_a_k(kappa) / kappa)))

    @override
    def carrier_measure(self, x: JaxComplexArray) -> JaxRealArray:
        xp = array_namespace(self, x)
        return xp.zeros(self.shape)

    @override
    @classmethod
    def sufficient_statistics(
        cls, x: JaxComplexArray, **fixed_parameters: JaxArray
    ) -> ComplexVonMisesEP:
        return ComplexVonMisesEP(x)

    def kappa(self) -> JaxRealArray:
        xp = array_namespace(self)
        return xp.abs(self.mean_times_concentration)

    def angle(self) -> JaxRealArray:
        xp = array_namespace(self)
        return xp.angle(self.mean_times_concentration)


@dataclass
class ComplexVonMisesEP(
    HasEntropyEP[ComplexVonMisesNP], ExpToNat[ComplexVonMisesNP], SimpleDistribution
):
    """The expectation parametrization of the von Mises distribution.

    Args:
        mean: E(exp(i x)).
    """

    mean: JaxComplexArray = distribution_parameter(
        ScalarSupport(ring=ComplexField(maximum_modulus=1.0))
    )

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
    def natural_parametrization_cls(cls) -> type[ComplexVonMisesNP]:
        return ComplexVonMisesNP

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = array_namespace(self)
        return xp.zeros(self.shape)

    @override
    def initial_search_parameters(self) -> JaxRealArray:
        xp = array_namespace(self)
        mu = xp.abs(self.mean)
        initial_kappa = xp.where(mu == 1.0, xp.asarray(xp.inf), (2.0 * mu - mu**3) / (1.0 - mu**2))
        initial_kappa = cast("JaxRealArray", initial_kappa)
        return inverse_softplus(initial_kappa)[..., xp.newaxis]

    @override
    def search_to_natural(self, search_parameters: JaxRealArray) -> ComplexVonMisesNP:
        xp = array_namespace(self)
        kappa = softplus(search_parameters[..., 0])
        mu = xp.abs(self.mean)
        q = xp.where(
            mu == 0.0,
            self.mean,
            self.mean * (kappa / mu),
        )
        return ComplexVonMisesNP(q)

    @override
    def search_gradient(self, search_parameters: JaxRealArray) -> JaxRealArray:
        xp = array_namespace(self)
        kappa = softplus(search_parameters[..., 0])
        mu = xp.abs(self.mean)
        return (_a_k(kappa) - mu)[..., xp.newaxis]


def _a_k(kappa: JaxRealArray) -> JaxRealArray:
    return jss.i1e(kappa) / jss.i0e(kappa)
