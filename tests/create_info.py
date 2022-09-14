from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import scipy.stats as ss
from jax.random import KeyArray
from tjax import ComplexArray, RealArray, Shape

from efax import (BernoulliEP, BernoulliNP, BetaEP, BetaNP, ChiEP, ChiNP, ChiSquareEP, ChiSquareNP,
                  ComplexCircularlySymmetricNormalEP, ComplexCircularlySymmetricNormalNP,
                  ComplexMultivariateUnitNormalEP, ComplexMultivariateUnitNormalNP, ComplexNormalEP,
                  ComplexNormalNP, DirichletEP, DirichletNP, ExponentialEP, ExponentialNP, GammaEP,
                  GammaNP, GeneralizedDirichletEP, GeneralizedDirichletNP, GeometricEP, GeometricNP,
                  IsotropicNormalEP, IsotropicNormalNP, LogarithmicEP, LogarithmicNP,
                  MultivariateDiagonalNormalEP, MultivariateDiagonalNormalNP,
                  MultivariateFixedVarianceNormalEP, MultivariateFixedVarianceNormalNP,
                  MultivariateNormalEP, MultivariateUnitNormalEP, MultivariateUnitNormalNP,
                  NegativeBinomialEP, NegativeBinomialNP, NormalEP, NormalNP, PoissonEP, PoissonNP,
                  RayleighEP, RayleighNP, ScipyComplexMultivariateNormal, ScipyComplexNormal,
                  ScipyDirichlet, ScipyGeneralizedDirichlet, ScipyMultivariateNormal, ScipyVonMises,
                  VonMisesFisherEP, VonMisesFisherNP, WeibullEP, WeibullNP)
from efax._src.tools import create_diagonal, np_abs_square, vectorized_tril, vectorized_triu

from .distribution_info import DistributionInfo


def dirichlet_parameter_generator(n: int, rng: KeyArray, shape: Shape) -> RealArray:
    # q can be as low as -1, but we prevent low values
    return rng.exponential(size=(*shape, n), scale=4.0) + 0.7


def generate_real_covariance(rng: KeyArray, dimensions: int) -> RealArray:
    if dimensions == 1:
        return np.ones((1, 1)) * rng.exponential()
    eigenvalues = rng.exponential(size=dimensions) + 1.0
    eigenvalues /= np.mean(eigenvalues)
    return ss.random_correlation.rvs(eigenvalues, random_state=rng)


def vectorized_real_covariance(rng: KeyArray, shape: Shape, dimensions: int) -> RealArray:
    if shape == ():
        return generate_real_covariance(rng, dimensions)
    return np.array([vectorized_real_covariance(rng, shape[1:], dimensions)
                     for _ in range(shape[0])])


def generate_complex_covariance(rng: KeyArray, dimensions: int) -> ComplexArray:
    x = generate_real_covariance(rng, dimensions)
    if dimensions == 1:
        return x
    y = generate_real_covariance(rng, dimensions)
    w = x + 1j * y
    return w @ (w.conjugate().T)  # type: ignore[return-value]


def vectorized_complex_covariance(rng: KeyArray, shape: Shape, dimensions: int) -> ComplexArray:
    if shape == ():
        return generate_complex_covariance(rng, dimensions)
    return np.array([vectorized_complex_covariance(rng, shape[1:], dimensions)
                     for _ in range(shape[0])])


class BernoulliInfo(DistributionInfo[BernoulliNP, BernoulliEP, RealArray]):
    def exp_to_scipy_distribution(self, p: BernoulliEP) -> Any:
        return ss.bernoulli(p.probability)

    def exp_parameter_generator(self, rng: KeyArray, shape: Shape) -> BernoulliEP:
        return BernoulliEP(rng.uniform(size=shape))


class GeometricInfo(DistributionInfo[GeometricNP, GeometricEP, RealArray]):
    def exp_to_scipy_distribution(self, p: GeometricEP) -> Any:
        # p is inverse odds
        return ss.geom(1.0 / (1.0 + p.mean))

    def exp_parameter_generator(self, rng: KeyArray, shape: Shape) -> GeometricEP:
        return GeometricEP(rng.exponential(size=shape))

    def scipy_to_exp_family_observation(self, x: RealArray) -> RealArray:
        return x - 1


class PoissonInfo(DistributionInfo[PoissonNP, PoissonEP, RealArray]):
    def exp_to_scipy_distribution(self, p: PoissonEP) -> Any:
        return ss.poisson(p.mean)

    def exp_parameter_generator(self, rng: KeyArray, shape: Shape) -> PoissonEP:
        return PoissonEP(rng.exponential(size=shape))


class NegativeBinomialInfo(DistributionInfo[NegativeBinomialNP, NegativeBinomialEP, RealArray]):
    def __init__(self, r: int):
        super().__init__()
        self.r = r

    def exp_to_scipy_distribution(self, p: NegativeBinomialEP) -> Any:
        return ss.nbinom(self.r, 1.0 / (1.0 + p.mean / p.failures))

    def exp_parameter_generator(self, rng: KeyArray, shape: Shape) -> NegativeBinomialEP:
        return NegativeBinomialEP(self.r * np.ones(shape, dtype=jnp.int_),
                                  rng.exponential(size=shape))


class LogarithmicInfo(DistributionInfo[LogarithmicNP, LogarithmicEP, RealArray]):
    def nat_to_scipy_distribution(self, q: LogarithmicNP) -> Any:
        return ss.logser(np.exp(q.log_probability))

    def exp_parameter_generator(self, rng: KeyArray, shape: Shape) -> LogarithmicEP:
        return LogarithmicEP(rng.exponential(size=shape) + 1.0)


class NormalInfo(DistributionInfo[NormalNP, NormalEP, RealArray]):
    def exp_to_scipy_distribution(self, p: NormalEP) -> Any:
        return ss.norm(p.mean, np.sqrt(p.variance()))

    def exp_parameter_generator(self, rng: KeyArray, shape: Shape) -> NormalEP:
        mean = rng.normal(scale=4.0, size=shape)
        variance = rng.exponential(size=shape)
        return NormalEP(mean, mean ** 2 + variance)


class MultivariateFixedVarianceNormalInfo(DistributionInfo[MultivariateFixedVarianceNormalNP,
                                                           MultivariateFixedVarianceNormalEP,
                                                           RealArray]):
    def __init__(self, dimensions: int, variance: float):
        super().__init__()
        self.dimensions = dimensions
        self.variance = variance

    def exp_to_scipy_distribution(self, p: MultivariateFixedVarianceNormalEP) -> Any:
        cov = np.tile(np.eye(p.dimensions()), p.shape + (1, 1))
        for i in np.ndindex(*p.shape):
            cov[i] *= p.variance[i]
        return ScipyMultivariateNormal.from_mc(mean=p.mean, cov=cov)

    def exp_parameter_generator(self, rng: KeyArray, shape: Shape
                                ) -> MultivariateFixedVarianceNormalEP:
        variance = self.variance * jnp.ones(shape)
        return MultivariateFixedVarianceNormalEP(rng.normal(size=(*shape, self.dimensions)),
                                                 variance=variance)


class MultivariateUnitNormalInfo(DistributionInfo[MultivariateUnitNormalNP,
                                                  MultivariateUnitNormalEP,
                                                  RealArray]):
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def exp_to_scipy_distribution(self, p: MultivariateUnitNormalEP) -> Any:
        return ScipyMultivariateNormal.from_mc(mean=p.mean)

    def exp_parameter_generator(self, rng: KeyArray, shape: Shape) -> MultivariateUnitNormalEP:
        return MultivariateUnitNormalEP(rng.normal(size=(*shape, self.dimensions)))


class IsotropicNormalInfo(DistributionInfo[IsotropicNormalNP, IsotropicNormalEP, RealArray]):
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def exp_to_scipy_distribution(self, p: IsotropicNormalEP) -> Any:
        v = p.variance()
        e = np.eye(self.dimensions)
        return ScipyMultivariateNormal.from_mc(mean=p.mean, cov=np.multiply.outer(v, e))

    def exp_parameter_generator(self, rng: KeyArray, shape: Shape) -> IsotropicNormalEP:
        mean = rng.normal(size=(*shape, self.dimensions))
        total_variance = self.dimensions * rng.exponential(size=shape)
        return IsotropicNormalEP(mean, np.sum(np.square(mean)) + total_variance)


class MultivariateDiagonalNormalInfo(DistributionInfo[MultivariateDiagonalNormalNP,
                                                      MultivariateDiagonalNormalEP,
                                                      RealArray]):
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def exp_to_scipy_distribution(self, p: MultivariateDiagonalNormalEP) -> Any:
        return ScipyMultivariateNormal.from_mc(mean=p.mean, cov=create_diagonal(p.variance()))

    def exp_parameter_generator(self, rng: KeyArray, shape: Shape) -> MultivariateDiagonalNormalEP:
        dist_shape = (*shape, self.dimensions)
        mean = rng.normal(size=dist_shape)
        variance = rng.exponential(size=dist_shape)
        return MultivariateDiagonalNormalEP(mean, np.square(mean) + variance)


class MultivariateNormalInfo(DistributionInfo[MultivariateUnitNormalNP, MultivariateNormalEP,
                                              RealArray]):
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def exp_to_scipy_distribution(self, p: MultivariateNormalEP) -> Any:
        # Correct numerical errors introduced by various conversions.
        v = p.variance()
        v_transpose = v.swapaxes(-1, -2)
        covariance = vectorized_tril(v) + vectorized_triu(v_transpose, 1)
        return ScipyMultivariateNormal.from_mc(mean=p.mean, cov=covariance)

    def exp_parameter_generator(self, rng: KeyArray, shape: Shape) -> MultivariateNormalEP:
        covariance = vectorized_real_covariance(rng, shape, self.dimensions)
        mean = rng.normal(size=(*shape, self.dimensions))
        second_moment = covariance + mean[..., :, np.newaxis] * mean[..., np.newaxis, :]
        return MultivariateNormalEP(mean, second_moment)


class ComplexNormalInfo(DistributionInfo[ComplexNormalNP, ComplexNormalEP, ComplexArray]):
    def exp_to_scipy_distribution(self, p: ComplexNormalEP) -> Any:
        return ScipyComplexNormal(p.mean,
                                  p.second_moment - np_abs_square(p.mean),
                                  p.pseudo_second_moment - np.square(p.mean))

    def exp_parameter_generator(self, rng: KeyArray, shape: Shape) -> ComplexNormalEP:
        mean = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        variance = rng.exponential(size=shape)
        second_moment = np_abs_square(mean) + variance
        pseudo_variance = (variance * rng.beta(2, 2, size=shape)
                           * np.exp(1j * rng.uniform(0, 2 * np.pi, size=shape)))
        pseudo_second_moment = np.square(mean) + pseudo_variance
        return ComplexNormalEP(mean, second_moment, pseudo_second_moment)


class ComplexMultivariateUnitNormalInfo(DistributionInfo[ComplexMultivariateUnitNormalNP,
                                                         ComplexMultivariateUnitNormalEP,
                                                         ComplexArray]):
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def exp_to_scipy_distribution(self, p: ComplexMultivariateUnitNormalEP) -> Any:
        return ScipyComplexMultivariateNormal(mean=p.mean)

    def exp_parameter_generator(self,
                                rng: KeyArray,
                                shape: Shape) -> ComplexMultivariateUnitNormalEP:
        a = rng.normal(size=(*shape, self.dimensions))
        b = rng.normal(size=(*shape, self.dimensions))
        return ComplexMultivariateUnitNormalEP(a + 1j * b)


class ComplexCircularlySymmetricNormalInfo(DistributionInfo[ComplexCircularlySymmetricNormalNP,
                                                            ComplexCircularlySymmetricNormalEP,
                                                            ComplexArray]):
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def exp_to_scipy_distribution(self, p: ComplexCircularlySymmetricNormalEP) -> Any:
        return ScipyComplexMultivariateNormal(variance=p.variance)

    def exp_parameter_generator(self,
                                rng: KeyArray,
                                shape: Shape) -> ComplexCircularlySymmetricNormalEP:
        return ComplexCircularlySymmetricNormalEP(vectorized_complex_covariance(rng, shape,
                                                                                self.dimensions))


class ExponentialInfo(DistributionInfo[ExponentialNP, ExponentialEP, RealArray]):
    def exp_to_scipy_distribution(self, p: ExponentialEP) -> Any:
        return ss.expon(0, p.mean)

    def exp_parameter_generator(self, rng: KeyArray, shape: Shape) -> ExponentialEP:
        return ExponentialEP(rng.exponential(size=shape))


class RayleighInfo(DistributionInfo[RayleighNP, RayleighEP, RealArray]):
    def exp_to_scipy_distribution(self, p: RayleighEP) -> Any:
        return ss.rayleigh(scale=np.sqrt(p.chi / 2.0))

    def exp_parameter_generator(self, rng: KeyArray, shape: Shape) -> RayleighEP:
        return RayleighEP(rng.exponential(size=shape))


class BetaInfo(DistributionInfo[BetaNP, BetaEP, RealArray]):
    def nat_to_scipy_distribution(self, q: BetaNP) -> Any:
        n1 = q.alpha_minus_one + 1.0
        return ss.beta(n1[..., 0], n1[..., 1])

    def nat_parameter_generator(self, rng: KeyArray, shape: Shape) -> BetaNP:
        return BetaNP(dirichlet_parameter_generator(2, rng, shape))


class GammaInfo(DistributionInfo[GammaNP, GammaEP, RealArray]):
    def nat_to_scipy_distribution(self, q: GammaNP) -> Any:
        shape = q.shape_minus_one + 1.0
        scale = -1.0 / q.negative_rate
        return ss.gamma(shape, scale=scale)

    def nat_parameter_generator(self, rng: KeyArray, shape: Shape) -> GammaNP:
        gamma_shape = rng.exponential(size=shape)
        rate = rng.exponential(size=shape)
        return GammaNP(-rate, gamma_shape - 1.0)


class DirichletInfo(DistributionInfo[DirichletNP, DirichletEP, RealArray]):
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def nat_to_scipy_distribution(self, q: DirichletNP) -> Any:
        return ScipyDirichlet(q.alpha_minus_one + 1.0)

    def nat_parameter_generator(self, rng: KeyArray, shape: Shape) -> DirichletNP:
        return DirichletNP(dirichlet_parameter_generator(self.dimensions, rng, shape))

    def scipy_to_exp_family_observation(self, x: RealArray) -> RealArray:
        return x[..., : -1]


class GeneralizedDirichletInfo(DistributionInfo[GeneralizedDirichletNP, GeneralizedDirichletEP,
                                                RealArray]):
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def nat_to_scipy_distribution(self, q: GeneralizedDirichletNP) -> Any:
        return ScipyGeneralizedDirichlet(*q.alpha_beta())

    def nat_parameter_generator(self, rng: KeyArray, shape: Shape) -> GeneralizedDirichletNP:
        alpha_minus_one = dirichlet_parameter_generator(self.dimensions, rng, shape)
        gamma = dirichlet_parameter_generator(self.dimensions, rng, shape) + 1.0
        return GeneralizedDirichletNP(alpha_minus_one, gamma)


class VonMisesFisherInfo(DistributionInfo[VonMisesFisherNP, VonMisesFisherEP, RealArray]):
    def nat_to_scipy_distribution(self, q: VonMisesFisherNP) -> Any:
        return ScipyVonMises(*q.to_kappa_angle())

    def nat_parameter_generator(self, rng: KeyArray, shape: Shape) -> VonMisesFisherNP:
        return VonMisesFisherNP(rng.normal(size=(*shape, 2), scale=4.0))

    def scipy_to_exp_family_observation(self, x: RealArray) -> RealArray:
        x = np.asarray(x)
        result = np.empty(x.shape + (2,))
        result[..., 0] = np.cos(x)
        result[..., 1] = np.sin(x)
        return result


class ChiSquareInfo(DistributionInfo[ChiSquareNP, ChiSquareEP, RealArray]):
    def nat_to_scipy_distribution(self, q: ChiSquareNP) -> Any:
        return ss.chi2((q.k_over_two_minus_one + 1.0) * 2.0)

    def nat_parameter_generator(self, rng: KeyArray, shape: Shape) -> ChiSquareNP:
        return ChiSquareNP(rng.exponential(size=shape))


class ChiInfo(DistributionInfo[ChiNP, ChiEP, RealArray]):
    def nat_to_scipy_distribution(self, q: ChiNP) -> Any:
        return ss.chi((q.k_over_two_minus_one + 1.0) * 2.0)

    def nat_parameter_generator(self, rng: KeyArray, shape: Shape) -> ChiNP:
        return ChiNP(rng.exponential(size=shape))


class WeibullInfo(DistributionInfo[WeibullNP, WeibullEP, RealArray]):
    def exp_to_scipy_distribution(self, p: WeibullEP) -> Any:
        scale = p.chi ** (1.0 / p.concentration)
        return ss.weibull_min(p.concentration, scale=scale)

    def nat_parameter_generator(self, rng: KeyArray, shape: Shape) -> WeibullNP:
        equal_fixed_parameters = True
        concentration = (np.broadcast_to(rng.exponential(), shape)
                         if equal_fixed_parameters
                         else rng.exponential(size=shape)) + 1.0
        return WeibullNP(concentration, -rng.exponential(size=shape) - 1.0)


def create_infos() -> list[DistributionInfo[Any, Any, Any]]:
    # pylint: disable=too-many-locals
    # Discrete
    bernoulli = BernoulliInfo()
    geometric = GeometricInfo()
    poisson = PoissonInfo()
    negative_binomial = NegativeBinomialInfo(3)
    logarithmic = LogarithmicInfo()
    discrete: list[DistributionInfo[Any, Any, Any]] = [bernoulli, geometric, poisson,
                                                       negative_binomial, logarithmic]

    # Continuous
    normal = NormalInfo()
    complex_normal = ComplexNormalInfo()
    cmvn_unit = ComplexMultivariateUnitNormalInfo(dimensions=4)
    cmvn_cs = ComplexCircularlySymmetricNormalInfo(dimensions=3)
    exponential = ExponentialInfo()
    rayleigh = RayleighInfo()
    gamma = GammaInfo()
    beta = BetaInfo()
    dirichlet = DirichletInfo(5)
    gen_dirichlet = GeneralizedDirichletInfo(5)
    von_mises = VonMisesFisherInfo()
    chi_square = ChiSquareInfo()
    chi = ChiInfo()
    weibull = WeibullInfo()
    continuous: list[DistributionInfo[Any, Any, Any]] = [normal, complex_normal, cmvn_unit, cmvn_cs,
                                                         exponential, rayleigh, gamma, beta,
                                                         dirichlet, gen_dirichlet, von_mises,
                                                         chi_square, chi, weibull]

    # Multivariate normal
    multivariate_fixed_variance_normal = MultivariateFixedVarianceNormalInfo(dimensions=5,
                                                                             variance=3.0)
    multivariate_unit_normal = MultivariateUnitNormalInfo(dimensions=5)
    isotropic_normal = IsotropicNormalInfo(dimensions=4)
    diagonal_normal = MultivariateDiagonalNormalInfo(dimensions=4)
    multivariate_normal = MultivariateNormalInfo(dimensions=4)
    mvn: list[DistributionInfo[Any, Any, Any]] = [multivariate_fixed_variance_normal,
                                                  multivariate_unit_normal, isotropic_normal,
                                                  diagonal_normal, multivariate_normal]

    return discrete + continuous + mvn
