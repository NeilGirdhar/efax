from __future__ import annotations

import math
from typing import Any, override

import jax.numpy as jnp
import jax.scipy.special as jss
from array_api_compat import array_namespace
from tjax import JaxArray, JaxRealArray, Shape, softplus
from tjax.dataclasses import dataclass

from efax._src.interfaces.multidimensional import Multidimensional
from efax._src.mixins.exp_to_nat.exp_to_nat import ExpToNat
from efax._src.mixins.has_entropy import HasEntropyEP, HasEntropyNP
from efax._src.natural_parametrization import NaturalParametrization
from efax._src.parameter import (
    RealField,
    ScalarSupport,
    Support,
    SymmetricMatrixSupport,
    distribution_parameter,
)


def _multivariate_digamma_sum(half_df: JaxRealArray, dimensions: int) -> JaxRealArray:
    return sum((jss.digamma(half_df - 0.5 * i) for i in range(dimensions)), jnp.zeros(()))


def _minimum_df_offset(dimensions: int) -> float:
    del dimensions
    return -1.0


@dataclass
class WishartNP(
    HasEntropyNP["WishartEP"],
    NaturalParametrization["WishartEP", JaxRealArray],
    Multidimensional,
):
    """The natural parametrization of the Wishart distribution.

    Args:
        negative_half_precision: ``-0.5 * scale^{-1}``.
        df_offset: ``(df - p - 1) / 2`` where ``p`` is the matrix dimension.
    """

    negative_half_precision: JaxRealArray = distribution_parameter(
        SymmetricMatrixSupport(negative_semidefinite=True)
    )
    df_offset: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.df_offset.shape

    @override
    @classmethod
    def domain_support(cls) -> SymmetricMatrixSupport:
        return SymmetricMatrixSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = array_namespace(self)
        half_df = self.half_df()
        _, logdet_precision = xp.linalg.slogdet(self.precision())
        return (
            -half_df * logdet_precision
            + half_df * self.dimensions() * math.log(2.0)
            + jss.multigammaln(half_df, self.dimensions())
        )

    @override
    def to_exp(self) -> WishartEP:
        xp = array_namespace(self)
        half_df = self.half_df()
        precision = self.precision()
        scale = xp.linalg.inv(precision)
        _, logdet_precision = xp.linalg.slogdet(precision)
        mean = 2.0 * half_df[..., xp.newaxis, xp.newaxis] * scale
        expected_log_det = (
            _multivariate_digamma_sum(half_df, self.dimensions())
            + self.dimensions() * math.log(2.0)
            - logdet_precision
        )
        return WishartEP(mean, expected_log_det)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = array_namespace(self, x)
        return xp.zeros(x.shape[:-2])

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any) -> WishartEP:
        xp = array_namespace(x)
        _, logdet_x = xp.linalg.slogdet(x)
        return WishartEP(x, logdet_x)

    @override
    def dimensions(self) -> int:
        return self.negative_half_precision.shape[-1]

    @override
    @classmethod
    def adjust_support(cls, support: Support, name: str, **kwargs: JaxArray) -> Support:
        if name != "df_offset":
            return super().adjust_support(support, name, **kwargs)
        p = kwargs["negative_half_precision"].shape[-1]
        return ScalarSupport(ring=RealField(minimum=_minimum_df_offset(p)))

    def precision(self) -> JaxRealArray:
        return -2.0 * self.negative_half_precision

    def half_df(self) -> JaxRealArray:
        return self.df_offset + 0.5 * (self.dimensions() + 1)


@dataclass
class WishartEP(HasEntropyEP[WishartNP], ExpToNat[WishartNP], Multidimensional):
    """The expectation parametrization of the Wishart distribution.

    Args:
        mean: ``E[X]``.
        expected_log_det: ``E[log |X|]``.
    """

    mean: JaxRealArray = distribution_parameter(SymmetricMatrixSupport(positive_semidefinite=True))
    expected_log_det: JaxRealArray = distribution_parameter(ScalarSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.expected_log_det.shape

    @override
    @classmethod
    def domain_support(cls) -> SymmetricMatrixSupport:
        return SymmetricMatrixSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[WishartNP]:
        return WishartNP

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = array_namespace(self)
        return xp.zeros(self.shape)

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    @override
    def initial_search_parameters(self) -> JaxRealArray:
        xp = array_namespace(self)
        return xp.zeros((*self.shape, 1))

    @override
    def search_to_natural(self, search_parameters: JaxRealArray) -> WishartNP:
        xp = array_namespace(self)
        df_offset = _minimum_df_offset(self.dimensions()) + softplus(search_parameters[..., 0])
        half_df = df_offset + 0.5 * (self.dimensions() + 1)
        precision = 2.0 * half_df[..., xp.newaxis, xp.newaxis] * xp.linalg.inv(self.mean)
        negative_half_precision = -0.5 * precision
        return WishartNP(negative_half_precision, df_offset)

    @override
    def search_gradient(self, search_parameters: JaxRealArray) -> JaxRealArray:
        xp = array_namespace(self)
        candidate = self.search_to_natural(search_parameters)
        return (candidate.to_exp().expected_log_det - self.expected_log_det)[..., xp.newaxis]
