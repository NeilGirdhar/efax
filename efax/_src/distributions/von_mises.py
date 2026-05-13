from __future__ import annotations

import math
from typing import cast, override

import jax
import jax.numpy as jnp
import jax.scipy.special as jss
from array_api_compat import array_namespace
from tjax import (
    JaxArray,
    JaxRealArray,
    Shape,
    bessel_iv_ratio,
    inverse_softplus,
    softplus,
)
from tjax.dataclasses import dataclass

from efax._src.interfaces.multidimensional import Multidimensional
from efax._src.mixins.exp_to_nat.exp_to_nat import ExpToNat
from efax._src.mixins.has_entropy import HasEntropyEP, HasEntropyNP
from efax._src.natural_parametrization import NaturalParametrization
from efax._src.parameter import CircularBoundedSupport, VectorSupport, distribution_parameter

_BESSEL_SERIES_TERMS = 200


@dataclass
class VonMisesFisherNP(
    HasEntropyNP["VonMisesFisherEP"],
    NaturalParametrization["VonMisesFisherEP", JaxRealArray],
    Multidimensional,
):
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
        xp = array_namespace(self)
        half_k = xp.asarray(self.dimensions() * 0.5)
        rho = xp.sum(xp.square(self.mean_times_concentration), axis=-1)
        return half_k * math.log(2.0 * math.pi) + _log_bessel_iv_over_power(half_k - 1.0, rho)

    @override
    def to_exp(self) -> VonMisesFisherEP:
        xp = array_namespace(self)
        q = self.mean_times_concentration
        kappa: JaxRealArray = xp.linalg.vector_norm(q, axis=-1, keepdims=True)
        return VonMisesFisherEP(
            xp.where(kappa == 0.0, q, q * (_a_k(xp.asarray(self.dimensions()), kappa) / kappa))
        )

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = array_namespace(self, x)
        return xp.zeros(self.shape)

    @override
    @classmethod
    def sufficient_statistics(
        cls, x: JaxRealArray, **fixed_parameters: JaxArray
    ) -> VonMisesFisherEP:
        return VonMisesFisherEP(x)

    @override
    def dimensions(self) -> int:
        return self.mean_times_concentration.shape[-1]

    def kappa(self) -> JaxRealArray:
        xp = array_namespace(self)
        return xp.linalg.vector_norm(self.mean_times_concentration, axis=-1)

    def to_kappa_angle(self) -> tuple[JaxRealArray, JaxRealArray]:
        if self.dimensions() != 2:  # noqa: PLR2004
            raise ValueError
        xp = array_namespace(self)
        kappa = self.kappa()
        angle = xp.where(
            kappa == 0.0,
            xp.asarray(0.0),
            xp.atan2(self.mean_times_concentration[..., 1], self.mean_times_concentration[..., 0]),
        )
        return kappa, angle


@dataclass
class VonMisesFisherEP(
    HasEntropyEP[VonMisesFisherNP], ExpToNat[VonMisesFisherNP], Multidimensional
):
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
        xp = array_namespace(self)
        return xp.zeros(self.shape)

    @override
    def initial_search_parameters(self) -> JaxRealArray:
        xp = array_namespace(self)
        mu: JaxRealArray = xp.linalg.vector_norm(self.mean, axis=-1)
        # 0 <= mu <= 1.0
        initial_kappa = xp.where(
            mu == 1.0, xp.asarray(xp.inf), (mu * self.dimensions() - mu**3) / (1.0 - mu**2)
        )
        initial_kappa = cast("JaxRealArray", initial_kappa)
        return inverse_softplus(initial_kappa)[..., xp.newaxis]

    @override
    def search_to_natural(self, search_parameters: JaxRealArray) -> VonMisesFisherNP:
        xp = array_namespace(self)
        kappa = softplus(search_parameters)
        mu = xp.linalg.vector_norm(self.mean, axis=-1, keepdims=True)
        q = self.mean * (kappa / mu)
        return VonMisesFisherNP(q)

    @override
    def search_gradient(self, search_parameters: JaxRealArray) -> JaxRealArray:
        xp = array_namespace(self)
        kappa = softplus(search_parameters)
        mu = xp.linalg.vector_norm(self.mean, axis=-1)
        return _a_k(xp.asarray(self.dimensions()), kappa) - mu

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]


# Private functions --------------------------------------------------------------------------------
def _a_k(k: JaxRealArray, kappa: JaxRealArray) -> JaxRealArray:
    return bessel_iv_ratio(k * 0.5, kappa)


def _log_bessel_iv_over_power(v: JaxRealArray, rho: JaxArray) -> JaxArray:
    """Return ``log(I_v(sqrt(rho)) / sqrt(rho)^v)`` by its rho-series."""
    rho = jnp.asarray(rho)
    rho_is_complex = jnp.iscomplexobj(rho)
    v, rho = jnp.broadcast_arrays(jnp.asarray(v), rho)
    dtype = jnp.result_type(rho, 1j)
    rho = rho.astype(dtype)
    v = v.astype(jnp.result_type(v, 1.0))
    term0 = jnp.exp(
        -v.astype(dtype) * jnp.log(jnp.asarray(2.0, dtype=dtype))
        - jss.gammaln(v + 1.0).astype(dtype)
    )

    def body(i: int, carry: tuple[JaxArray, JaxArray]) -> tuple[JaxArray, JaxArray]:
        term, total = carry
        i_array = jnp.asarray(i, dtype=v.dtype)
        term = term * rho / (4.0 * i_array * (i_array + v))
        return term, total + term

    _, total = jax.lax.fori_loop(1, _BESSEL_SERIES_TERMS, body, (term0, term0))
    result = jnp.log(total)
    return result if rho_is_complex else jnp.real(result)
