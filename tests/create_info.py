from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import scipy.stats as ss
from tjax import JaxRealArray, NumpyComplexArray, NumpyRealArray, abs_square, create_diagonal_array
from typing_extensions import override

from efax import (BernoulliEP, BernoulliNP, BetaEP, BetaNP, ChiEP, ChiNP, ChiSquareEP, ChiSquareNP,
                  ComplexCircularlySymmetricNormalEP, ComplexCircularlySymmetricNormalNP,
                  ComplexMultivariateUnitNormalEP, ComplexMultivariateUnitNormalNP, ComplexNormalEP,
                  ComplexNormalNP, ComplexUnitNormalEP, ComplexUnitNormalNP, DirichletEP,
                  DirichletNP, ExponentialEP, ExponentialNP, GammaEP, GammaNP,
                  GeneralizedDirichletEP, GeneralizedDirichletNP, GeometricEP, GeometricNP,
                  IsotropicNormalEP, IsotropicNormalNP, JointDistributionE, JointDistributionN,
                  LogarithmicEP, LogarithmicNP, MultivariateDiagonalNormalEP,
                  MultivariateDiagonalNormalNP, MultivariateFixedVarianceNormalEP,
                  MultivariateFixedVarianceNormalNP, MultivariateNormalEP, MultivariateNormalNP,
                  MultivariateUnitNormalEP, MultivariateUnitNormalNP, NegativeBinomialEP,
                  NegativeBinomialNP, NormalEP, NormalNP, PoissonEP, PoissonNP, RayleighEP,
                  RayleighNP, ScipyComplexMultivariateNormal, ScipyComplexNormal, ScipyDirichlet,
                  ScipyGeneralizedDirichlet, ScipyGeometric, ScipyJointDistribution,
                  ScipyMultivariateNormal, ScipyVonMises, ScipyVonMisesFisher, Structure,
                  SubDistributionInfo, UnitNormalEP, UnitNormalNP, VonMisesFisherEP,
                  VonMisesFisherNP, WeibullEP, WeibullNP)

from .distribution_info import DistributionInfo


class BernoulliInfo(DistributionInfo[BernoulliNP, BernoulliEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: BernoulliEP) -> Any:
        return ss.bernoulli(p.probability)

    @override
    def exp_class(self) -> type[BernoulliEP]:
        return BernoulliEP

    @override
    def nat_class(self) -> type[BernoulliNP]:
        return BernoulliNP


class GeometricInfo(DistributionInfo[GeometricNP, GeometricEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: GeometricEP) -> Any:
        # Scipy uses a different definition geometric distribution.  The parameter p is inverse
        # odds.
        return ScipyGeometric(np.asarray(1.0 / (1.0 + p.mean)))

    @override
    def scipy_to_exp_family_observation(self, x: NumpyRealArray) -> JaxRealArray:
        return jnp.asarray(x - 1)

    @override
    def exp_class(self) -> type[GeometricEP]:
        return GeometricEP

    @override
    def nat_class(self) -> type[GeometricNP]:
        return GeometricNP


class PoissonInfo(DistributionInfo[PoissonNP, PoissonEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: PoissonEP) -> Any:
        return ss.poisson(p.mean)

    @override
    def exp_class(self) -> type[PoissonEP]:
        return PoissonEP

    @override
    def nat_class(self) -> type[PoissonNP]:
        return PoissonNP


class NegativeBinomialInfo(DistributionInfo[NegativeBinomialNP, NegativeBinomialEP,
                                            NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: NegativeBinomialEP) -> Any:
        return ss.nbinom(p.failures, 1.0 / (1.0 + p.mean / p.failures))

    @override
    def exp_class(self) -> type[NegativeBinomialEP]:
        return NegativeBinomialEP

    @override
    def nat_class(self) -> type[NegativeBinomialNP]:
        return NegativeBinomialNP


class LogarithmicInfo(DistributionInfo[LogarithmicNP, LogarithmicEP, NumpyRealArray]):
    @override
    def nat_to_scipy_distribution(self, q: LogarithmicNP) -> Any:
        return ss.logser(np.exp(q.log_probability))

    @override
    def exp_class(self) -> type[LogarithmicEP]:
        return LogarithmicEP

    @override
    def nat_class(self) -> type[LogarithmicNP]:
        return LogarithmicNP


class NormalInfo(DistributionInfo[NormalNP, NormalEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: NormalEP) -> Any:
        return ss.norm(p.mean, np.sqrt(p.variance()))

    @override
    def exp_class(self) -> type[NormalEP]:
        return NormalEP

    @override
    def nat_class(self) -> type[NormalNP]:
        return NormalNP


class UnitNormalInfo(DistributionInfo[UnitNormalNP, UnitNormalEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: UnitNormalEP) -> Any:
        return ss.norm(p.mean, 1.0)

    @override
    def exp_class(self) -> type[UnitNormalEP]:
        return UnitNormalEP

    @override
    def nat_class(self) -> type[UnitNormalNP]:
        return UnitNormalNP


class MultivariateFixedVarianceNormalInfo(DistributionInfo[MultivariateFixedVarianceNormalNP,
                                                           MultivariateFixedVarianceNormalEP,
                                                           NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: MultivariateFixedVarianceNormalEP) -> Any:
        cov = np.tile(np.eye(p.dimensions()), (*p.shape, 1, 1))
        for i in np.ndindex(*p.shape):
            cov[i] *= p.variance[i]
        return ScipyMultivariateNormal.from_mc(mean=np.asarray(p.mean), cov=np.asarray(cov))

    @override
    def exp_class(self) -> type[MultivariateFixedVarianceNormalEP]:
        return MultivariateFixedVarianceNormalEP

    @override
    def nat_class(self) -> type[MultivariateFixedVarianceNormalNP]:
        return MultivariateFixedVarianceNormalNP


class MultivariateUnitNormalInfo(DistributionInfo[MultivariateUnitNormalNP,
                                                  MultivariateUnitNormalEP,
                                                  NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: MultivariateUnitNormalEP) -> Any:
        return ScipyMultivariateNormal.from_mc(mean=np.asarray(p.mean))

    @override
    def exp_class(self) -> type[MultivariateUnitNormalEP]:
        return MultivariateUnitNormalEP

    @override
    def nat_class(self) -> type[MultivariateUnitNormalNP]:
        return MultivariateUnitNormalNP


class IsotropicNormalInfo(DistributionInfo[IsotropicNormalNP, IsotropicNormalEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: IsotropicNormalEP) -> Any:
        v = p.variance()
        e = np.eye(self.dimensions)
        return ScipyMultivariateNormal.from_mc(mean=np.asarray(p.mean),
                                               cov=np.asarray(np.multiply.outer(v, e)))

    @override
    def exp_class(self) -> type[IsotropicNormalEP]:
        return IsotropicNormalEP

    @override
    def nat_class(self) -> type[IsotropicNormalNP]:
        return IsotropicNormalNP


class MultivariateDiagonalNormalInfo(DistributionInfo[MultivariateDiagonalNormalNP,
                                                      MultivariateDiagonalNormalEP,
                                                      NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: MultivariateDiagonalNormalEP) -> Any:
        variance = np.asarray(p.variance())
        covariance = create_diagonal_array(variance)
        return ScipyMultivariateNormal.from_mc(mean=np.asarray(p.mean), cov=covariance)

    @override
    def exp_class(self) -> type[MultivariateDiagonalNormalEP]:
        return MultivariateDiagonalNormalEP

    @override
    def nat_class(self) -> type[MultivariateDiagonalNormalNP]:
        return MultivariateDiagonalNormalNP


class MultivariateNormalInfo(DistributionInfo[MultivariateNormalNP, MultivariateNormalEP,
                                              NumpyRealArray]):
    def __init__(self, dimensions: int) -> None:
        super().__init__(dimensions=dimensions, safety=0.1)

    @override
    def exp_to_scipy_distribution(self, p: MultivariateNormalEP) -> Any:
        # Correct numerical errors introduced by various conversions.
        mean = np.asarray(p.mean, dtype=np.float64)
        v = np.asarray(p.variance(), dtype=np.float64)
        v_transpose = v.swapaxes(-1, -2)
        covariance = np.tril(v) + np.triu(v_transpose, 1)
        return ScipyMultivariateNormal.from_mc(mean=mean, cov=covariance)

    @override
    def exp_class(self) -> type[MultivariateNormalEP]:
        return MultivariateNormalEP

    @override
    def nat_class(self) -> type[MultivariateNormalNP]:
        return MultivariateNormalNP


class ComplexUnitNormalInfo(DistributionInfo[ComplexUnitNormalNP, ComplexUnitNormalEP,
                                             NumpyComplexArray]):
    @override
    def exp_to_scipy_distribution(self, p: ComplexUnitNormalEP) -> Any:
        mean = np.asarray(p.mean, dtype=np.complex128)
        variance = np.ones_like(np.real(mean))
        pseudo_variance = np.zeros_like(mean)
        return ScipyComplexNormal(mean, variance, pseudo_variance)

    @override
    def exp_class(self) -> type[ComplexUnitNormalEP]:
        return ComplexUnitNormalEP

    @override
    def nat_class(self) -> type[ComplexUnitNormalNP]:
        return ComplexUnitNormalNP


class ComplexNormalInfo(DistributionInfo[ComplexNormalNP, ComplexNormalEP, NumpyComplexArray]):
    @override
    def exp_to_scipy_distribution(self, p: ComplexNormalEP) -> Any:
        mean = np.asarray(p.mean, dtype=np.complex128)
        second_moment = np.asarray(p.second_moment, dtype=np.float64)
        pseudo_second_moment = np.asarray(p.pseudo_second_moment, dtype=np.complex128)
        return ScipyComplexNormal(mean,
                                  second_moment - abs_square(mean),
                                  pseudo_second_moment - np.square(mean))

    @override
    def exp_class(self) -> type[ComplexNormalEP]:
        return ComplexNormalEP

    @override
    def nat_class(self) -> type[ComplexNormalNP]:
        return ComplexNormalNP


class ComplexMultivariateUnitNormalInfo(DistributionInfo[ComplexMultivariateUnitNormalNP,
                                                         ComplexMultivariateUnitNormalEP,
                                                         NumpyComplexArray]):
    @override
    def exp_to_scipy_distribution(self, p: ComplexMultivariateUnitNormalEP) -> Any:
        return ScipyComplexMultivariateNormal(mean=np.asarray(p.mean))

    @override
    def exp_class(self) -> type[ComplexMultivariateUnitNormalEP]:
        return ComplexMultivariateUnitNormalEP

    @override
    def nat_class(self) -> type[ComplexMultivariateUnitNormalNP]:
        return ComplexMultivariateUnitNormalNP


class ComplexCircularlySymmetricNormalInfo(DistributionInfo[ComplexCircularlySymmetricNormalNP,
                                                            ComplexCircularlySymmetricNormalEP,
                                                            NumpyComplexArray]):
    @override
    def exp_to_scipy_distribution(self, p: ComplexCircularlySymmetricNormalEP) -> Any:
        return ScipyComplexMultivariateNormal(variance=np.asarray(p.variance))

    @override
    def exp_class(self) -> type[ComplexCircularlySymmetricNormalEP]:
        return ComplexCircularlySymmetricNormalEP

    @override
    def nat_class(self) -> type[ComplexCircularlySymmetricNormalNP]:
        return ComplexCircularlySymmetricNormalNP


class ExponentialInfo(DistributionInfo[ExponentialNP, ExponentialEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: ExponentialEP) -> Any:
        return ss.expon(0, p.mean)

    @override
    def exp_class(self) -> type[ExponentialEP]:
        return ExponentialEP

    @override
    def nat_class(self) -> type[ExponentialNP]:
        return ExponentialNP


class RayleighInfo(DistributionInfo[RayleighNP, RayleighEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: RayleighEP) -> Any:
        return ss.rayleigh(scale=np.sqrt(p.chi / 2.0))

    @override
    def exp_class(self) -> type[RayleighEP]:
        return RayleighEP

    @override
    def nat_class(self) -> type[RayleighNP]:
        return RayleighNP


class BetaInfo(DistributionInfo[BetaNP, BetaEP, NumpyRealArray]):
    def __init__(self) -> None:
        super().__init__(dimensions=2, safety=0.1)

    @override
    def nat_to_scipy_distribution(self, q: BetaNP) -> Any:
        n1 = q.alpha_minus_one + 1.0
        return ss.beta(n1[..., 0], n1[..., 1])

    @override
    def exp_class(self) -> type[BetaEP]:
        return BetaEP

    @override
    def nat_class(self) -> type[BetaNP]:
        return BetaNP


class GammaInfo(DistributionInfo[GammaNP, GammaEP, NumpyRealArray]):
    @override
    def nat_to_scipy_distribution(self, q: GammaNP) -> Any:
        shape = q.shape_minus_one + 1.0
        scale = -1.0 / q.negative_rate
        return ss.gamma(shape, scale=scale)

    @override
    def exp_class(self) -> type[GammaEP]:
        return GammaEP

    @override
    def nat_class(self) -> type[GammaNP]:
        return GammaNP


class DirichletInfo(DistributionInfo[DirichletNP, DirichletEP, NumpyRealArray]):
    def __init__(self, dimensions: int) -> None:
        super().__init__(dimensions=dimensions, safety=0.1)

    @override
    def nat_to_scipy_distribution(self, q: DirichletNP) -> Any:
        return ScipyDirichlet(np.asarray(q.alpha_minus_one, dtype=np.float64) + 1.0)

    @override
    def scipy_to_exp_family_observation(self, x: NumpyRealArray) -> JaxRealArray:
        return jnp.asarray(x[..., : -1])

    @override
    def exp_class(self) -> type[DirichletEP]:
        return DirichletEP

    @override
    def nat_class(self) -> type[DirichletNP]:
        return DirichletNP


class GeneralizedDirichletInfo(DistributionInfo[GeneralizedDirichletNP, GeneralizedDirichletEP,
                                                NumpyRealArray]):
    def __init__(self, dimensions: int) -> None:
        super().__init__(dimensions=dimensions, safety=3.0)

    @override
    def nat_to_scipy_distribution(self, q: GeneralizedDirichletNP) -> Any:
        alpha, beta = q.alpha_beta()
        return ScipyGeneralizedDirichlet(np.asarray(alpha), np.asarray(beta))

    @override
    def exp_class(self) -> type[GeneralizedDirichletEP]:
        return GeneralizedDirichletEP

    @override
    def nat_class(self) -> type[GeneralizedDirichletNP]:
        return GeneralizedDirichletNP


class VonMisesInfo(DistributionInfo[VonMisesFisherNP, VonMisesFisherEP, NumpyRealArray]):
    def __init__(self) -> None:
        super().__init__(dimensions=2)

    @override
    def nat_to_scipy_distribution(self, q: VonMisesFisherNP) -> Any:
        kappa, angle = q.to_kappa_angle()
        return ScipyVonMises(np.asarray(kappa), np.asarray(angle))

    @override
    def scipy_to_exp_family_observation(self, x: NumpyRealArray) -> JaxRealArray:
        result = np.empty((*x.shape, 2))
        result[..., 0] = np.cos(x)
        result[..., 1] = np.sin(x)
        return jnp.asarray(result)

    @override
    def exp_class(self) -> type[VonMisesFisherEP]:
        return VonMisesFisherEP

    @override
    def nat_class(self) -> type[VonMisesFisherNP]:
        return VonMisesFisherNP


class VonMisesFisherInfo(DistributionInfo[VonMisesFisherNP, VonMisesFisherEP, NumpyRealArray]):
    @override
    def nat_to_scipy_distribution(self, q: VonMisesFisherNP) -> Any:
        kappa = np.asarray(q.kappa())
        mu = np.asarray(q.mean_times_concentration) / kappa[..., np.newaxis]
        return ScipyVonMisesFisher(mu, kappa)

    @override
    def exp_class(self) -> type[VonMisesFisherEP]:
        return VonMisesFisherEP

    @override
    def nat_class(self) -> type[VonMisesFisherNP]:
        return VonMisesFisherNP


class ChiSquareInfo(DistributionInfo[ChiSquareNP, ChiSquareEP, NumpyRealArray]):
    @override
    def nat_to_scipy_distribution(self, q: ChiSquareNP) -> Any:
        return ss.chi2((q.k_over_two_minus_one + 1.0) * 2.0)

    @override
    def exp_class(self) -> type[ChiSquareEP]:
        return ChiSquareEP

    @override
    def nat_class(self) -> type[ChiSquareNP]:
        return ChiSquareNP


class ChiInfo(DistributionInfo[ChiNP, ChiEP, NumpyRealArray]):
    @override
    def nat_to_scipy_distribution(self, q: ChiNP) -> Any:
        return ss.chi((q.k_over_two_minus_one + 1.0) * 2.0)

    @override
    def exp_class(self) -> type[ChiEP]:
        return ChiEP

    @override
    def nat_class(self) -> type[ChiNP]:
        return ChiNP


class WeibullInfo(DistributionInfo[WeibullNP, WeibullEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: WeibullEP) -> Any:
        scale = p.chi ** (1.0 / p.concentration)
        return ss.weibull_min(p.concentration, scale=scale)

    @override
    def exp_class(self) -> type[WeibullEP]:
        return WeibullEP

    @override
    def nat_class(self) -> type[WeibullNP]:
        return WeibullNP


class JointInfo(DistributionInfo[JointDistributionN, JointDistributionE, dict[str, Any]]):
    def __init__(self, infos: Mapping[str, DistributionInfo[Any, Any, Any]]) -> None:
        super().__init__()
        self.infos = dict(infos)

    @override
    def nat_to_scipy_distribution(self, q: JointDistributionN) -> Any:
        return ScipyJointDistribution(
                {name: info.nat_to_scipy_distribution(q.sub_distributions()[name])
                 for name, info in self.infos.items()})

    @override
    def scipy_to_exp_family_observation(self, x: dict[str, Any]) -> dict[str, Any]:
        assert isinstance(x, dict)
        return {name: info.scipy_to_exp_family_observation(x[name])
                for name, info in self.infos.items()}

    @override
    def exp_structure(self) -> Structure[JointDistributionE]:
        infos = []
        for name, info in self.infos.items():
            infos.extend([SubDistributionInfo((*sub_info.path, name),
                                              sub_info.type_,
                                              sub_info.dimensions,
                                              sub_info.sub_distribution_names)
                          for sub_info in info.exp_structure().infos])
        infos.append(SubDistributionInfo((), self.exp_class(), self.dimensions,
                                         list(self.infos.keys())))
        return Structure(infos)

    @override
    def nat_structure(self) -> Structure[JointDistributionN]:
        infos = []
        for name, info in self.infos.items():
            infos.extend([SubDistributionInfo((*sub_info.path, name),
                                              sub_info.type_,
                                              sub_info.dimensions,
                                              sub_info.sub_distribution_names)
                          for sub_info in info.nat_structure().infos])
        infos.append(SubDistributionInfo((), self.nat_class(), self.dimensions,
                                         list(self.infos.keys())))
        return Structure(infos)

    @override
    def exp_class(self) -> type[JointDistributionE]:
        return JointDistributionE

    @override
    def nat_class(self) -> type[JointDistributionN]:
        return JointDistributionN


def create_infos() -> list[DistributionInfo[Any, Any, Any]]:
    return [
            BernoulliInfo(),
            BetaInfo(),
            ChiInfo(),
            ChiSquareInfo(),
            ComplexCircularlySymmetricNormalInfo(dimensions=3),
            ComplexMultivariateUnitNormalInfo(dimensions=4),
            ComplexNormalInfo(),
            ComplexUnitNormalInfo(),
            DirichletInfo(dimensions=5),
            ExponentialInfo(),
            GammaInfo(),
            GeneralizedDirichletInfo(dimensions=5),
            GeometricInfo(),
            IsotropicNormalInfo(dimensions=5),
            LogarithmicInfo(),
            MultivariateDiagonalNormalInfo(dimensions=4),
            MultivariateFixedVarianceNormalInfo(dimensions=2),
            MultivariateNormalInfo(dimensions=4),
            MultivariateUnitNormalInfo(dimensions=5),
            NegativeBinomialInfo(),
            JointInfo(infos={'gamma': GammaInfo(), 'normal': NormalInfo()}),
            NormalInfo(),
            PoissonInfo(),
            RayleighInfo(),
            UnitNormalInfo(),
            VonMisesInfo(),
            VonMisesFisherInfo(dimensions=5),
            WeibullInfo(),
            ]
