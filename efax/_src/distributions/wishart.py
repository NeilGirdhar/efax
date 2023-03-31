from __future__ import annotations

from typing import Any

import jax.scipy.special as jss
from array_api_compat import array_namespace
from tjax import JaxArray, JaxRealArray, Shape, softplus
from tjax.dataclasses import dataclass
from typing import override

from ..interfaces.multidimensional import Multidimensional
from ..mixins.exp_to_nat.exp_to_nat import ExpToNat
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import (RealField, ScalarSupport, Support, SymmetricMatrixSupport,
                         distribution_parameter)


@dataclass
class WishartNP(HasEntropyNP['WishartEP'],
                NaturalParametrization['WishartEP', JaxRealArray],
                Multidimensional):
    """The natural parameters of the Wishart distribution.

    Consider the dimension p, the degrees of freedom v, and the scale matrix Σ.

    Args:
        df_offset: (v - p - 1) / 2
        negative_half_precision: -0.5 sigma ** -1
    """
    negative_half_precision: JaxRealArray = distribution_parameter(SymmetricMatrixSupport())
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
        xp = self.array_namespace()
        p = self.dimensions()
        half_df = self.df_offset + 0.5 * (p + 1)
        precision = -2.0 * self.negative_half_precision
        return (half_df * xp.log(xp.linalg.det(precision))
                + half_df * p * xp.log(2)
                + sum(jss.gammaln(self.df_offset + 1.0 + 0.5 * i) for i in range(p)))

    @override
    def to_exp(self) -> WishartEP:
        xp = self.array_namespace()
        p = self.dimensions()
        precision = -2.0 * self.negative_half_precision
        expected_log_det = (sum(jss.digamma(self.df_offset + 1.0 + 0.5 * i) for i in range(p))
                            + xp.log(xp.linalg.det(precision)))
        mean = (-2.0 * self.df_offset - (p + 1)) * xp.linalg.inv(self.negative_half_precision)
        return WishartEP(mean, expected_log_det)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.zeros(self.shape)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> WishartEP:
        xp = array_namespace(x)
        return WishartEP(x, xp.log(xp.linalg.det(x)))

    @override
    def dimensions(self) -> int:
        return self.negative_half_precision.shape[-1]

    @override
    @classmethod
    def adjust_support(cls, name: str, **kwargs: JaxArray) -> Support:
        if name != 'df_offset':
            return super().adjust_support(name, **kwargs)
        p = kwargs['negative_half_precision'].shape[-1]
        return ScalarSupport(ring=RealField(minimum=cls._minimum_df_offset(p)))

    @classmethod
    def _minimum_df_offset(cls, dimensions: int) -> float:
        return 0.5 * (-dimensions - 1)


@dataclass
class WishartEP(HasEntropyEP[WishartNP],
                ExpToNat[WishartNP],
                Multidimensional):
    """The expectation parameters of the Wishart distribution.

    Args:
        expected_log_det
        mean
    """
    mean: JaxRealArray = distribution_parameter(SymmetricMatrixSupport())
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
        xp = self.array_namespace()
        return xp.zeros(self.shape)

    @override
    def dimensions(self) -> int:
        return self.mean.shape[-1]

    @override
    def initial_search_parameters(self) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.zeros(self.shape)

    @override
    def search_to_natural(self, search_parameters: JaxRealArray) -> WishartNP:
        xp = self.array_namespace()
        # TODO: factor out the inversion
        negative_half_precision = -0.5 * xp.linalg.inv(self.mean)
        df_offset = (WishartNP._minimum_df_offset(self.dimensions())  # noqa: SLF001
                     + softplus(search_parameters))
        return WishartNP(negative_half_precision, df_offset)
