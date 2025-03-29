from __future__ import annotations

import math
from typing import cast

from tjax import JaxArray, JaxRealArray, Shape, inverse_softplus, softplus
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.multidimensional import Multidimensional
from ..mixins.exp_to_nat.exp_to_nat import ExpToNat
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import CircularBoundedSupport, VectorSupport, distribution_parameter
from ..tools import iv_ratio, log_ive


@dataclass
class VonMisesFisherNP(HasEntropyNP['VonMisesFisherEP'],
                       NaturalParametrization['VonMisesFisherEP', JaxRealArray],
                       Multidimensional):
    """The natural parametrization of the von Mises-Fisher distribution.

    Args:
        mean_times_concentration: E(x) times the concentration kappa.
    """
    mean_times_concentration: JaxRealArray = distribution_parameter(VectorSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean_times_concentration.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.array_namespace()
        half_k = self.dimensions() * 0.5
        kappa = xp.linalg.vector_norm(self.mean_times_concentration, axis=-1)
        return (kappa
                - (half_k - 1.0) * xp.log(kappa)
                + half_k * math.log(2.0 * math.pi)
                + log_ive(half_k - 1.0, kappa))

    @override
    def to_exp(self) -> VonMisesFisherEP:
        xp = self.array_namespace()
        q = self.mean_times_concentration
        kappa: JaxRealArray = xp.linalg.vector_norm(q, axis=-1, keepdims=True)
        return VonMisesFisherEP(
                xp.where(kappa == 0.0,
                         q,
                         q * (_a_k(self.dimensions(), kappa) / kappa)))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        return xp.zeros(self.shape)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray
                              ) -> VonMisesFisherEP:
        return VonMisesFisherEP(x)

    @override
    def dimensions(self) -> int:
        return self.mean_times_concentration.shape[-1]

    def kappa(self) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.linalg.vector_norm(self.mean_times_concentration, axis=-1)

    def to_kappa_angle(self) -> tuple[JaxRealArray, JaxRealArray]:
        if self.dimensions() != 2:  # noqa: PLR2004
            raise ValueError
        xp = self.array_namespace()
        kappa = self.kappa()
        angle = xp.where(kappa == 0.0,
                         xp.asarray(0.0),
                         xp.atan2(self.mean_times_concentration[..., 1],
                                  self.mean_times_concentration[..., 0]))
        return kappa, angle


@dataclass
class VonMisesFisherEP(HasEntropyEP[VonMisesFisherNP],
                       ExpToNat[VonMisesFisherNP],
                       Multidimensional):
    """The expectation parametrization of the von Mises-Fisher distribution.

    Args:
        mean: E(x).
    """
    mean: JaxRealArray = distribution_parameter(CircularBoundedSupport(1.0))

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape[:-1]

    @override
    @classmethod
    def domain_support(cls) -> VectorSupport:
        return VectorSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[VonMisesFisherNP]:
        return VonMisesFisherNP

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.zeros(self.shape)

    @override
    def initial_search_parameters(self) -> JaxRealArray:
        xp = self.array_namespace()
        mu: JaxRealArray = xp.linalg.vector_norm(self.mean, axis=-1)
        # 0 <= mu <= 1.0
        initial_kappa = xp.where(mu == 1.0,
                                 xp.asarray(xp.inf),
                                 (mu * self.dimensions() - mu ** 3) / (1.0 - mu ** 2))
        initial_kappa = cast('JaxRealArray', initial_kappa)
        return inverse_softplus(initial_kappa)[..., xp.newaxis]

    @override
    def search_to_natural(self, search_parameters: JaxRealArray) -> VonMisesFisherNP:
        xp = self.array_namespace()
        kappa = softplus(search_parameters)
        mu = xp.linalg.vector_norm(self.mean, axis=-1, keepdims=True)
        q = self.mean * (kappa / mu)
        return VonMisesFisherNP(q)

    @override
    def search_gradient(self, search_parameters: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace()
        kappa = softplus(search_parameters)
        mu = xp.linalg.vector_norm(self.mean, axis=-1)
        return _a_k(self.dimensions(), kappa) - mu

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]


# Private functions --------------------------------------------------------------------------------
def _a_k(k: float | JaxRealArray, kappa: float | JaxRealArray) -> JaxRealArray:
    return iv_ratio(k * 0.5, kappa)
