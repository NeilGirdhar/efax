from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import scipy.stats as ss
from numpy.random import Generator
from tjax import NumpyComplexArray, NumpyRealArray, Shape
from typing_extensions import override

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

int_dtype = jnp.asarray(1).dtype


def dirichlet_parameter_generator(n: int, rng: Generator, shape: Shape) -> NumpyRealArray:
    # q can be as low as -1, but we prevent low values
    return rng.exponential(size=(*shape, n), scale=4.0) + 0.7


def generate_real_covariance(rng: Generator, dimensions: int) -> NumpyRealArray:
    if dimensions == 1:
        return np.ones((1, 1)) * rng.exponential()
    eigenvalues = rng.exponential(size=dimensions) + 1.0
    eigenvalues /= np.mean(eigenvalues)
    return ss.random_correlation.rvs(eigenvalues, random_state=rng)


def vectorized_real_covariance(rng: Generator, shape: Shape, dimensions: int) -> NumpyRealArray:
    if shape == ():
        return generate_real_covariance(rng, dimensions)
    return np.asarray([vectorized_real_covariance(rng, shape[1:], dimensions)
                       for _ in range(shape[0])])


def generate_complex_covariance(rng: Generator, dimensions: int) -> NumpyComplexArray:
    x = generate_real_covariance(rng, dimensions)
    if dimensions == 1:
        return x
    y = generate_real_covariance(rng, dimensions)
    w: NumpyComplexArray = x + 1j * y
    return w @ (w.conjugate().T)


def vectorized_complex_covariance(rng: Generator, shape: Shape, dimensions: int
                                  ) -> NumpyComplexArray:
    if shape == ():
        return generate_complex_covariance(rng, dimensions)
    return np.asarray([vectorized_complex_covariance(rng, shape[1:], dimensions)
                       for _ in range(shape[0])])


class BernoulliInfo(DistributionInfo[BernoulliNP, BernoulliEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: BernoulliEP) -> Any:
        return ss.bernoulli(p.probability)

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> BernoulliEP:
        return BernoulliEP(jnp.asarray(rng.uniform(size=shape)))


class GeometricInfo(DistributionInfo[GeometricNP, GeometricEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: GeometricEP) -> Any:
        # p is inverse odds
        return ss.geom(1.0 / (1.0 + p.mean))

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> GeometricEP:
        return GeometricEP(jnp.asarray(rng.exponential(size=shape)))

    @override
    def scipy_to_exp_family_observation(self, x: NumpyRealArray) -> NumpyRealArray:
        return x - 1


class PoissonInfo(DistributionInfo[PoissonNP, PoissonEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: PoissonEP) -> Any:
        return ss.poisson(p.mean)

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> PoissonEP:
        return PoissonEP(jnp.asarray(rng.exponential(size=shape)))


class NegativeBinomialInfo(DistributionInfo[NegativeBinomialNP, NegativeBinomialEP,
                                            NumpyRealArray]):
    @override
    def __init__(self, r: int):
        super().__init__()
        self.r = r

    @override
    def exp_to_scipy_distribution(self, p: NegativeBinomialEP) -> Any:
        return ss.nbinom(self.r, 1.0 / (1.0 + p.mean / p.failures))

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> NegativeBinomialEP:
        return NegativeBinomialEP(jnp.asarray(rng.exponential(size=shape)),
                                  self.r * jnp.ones(shape, dtype=int_dtype))


class LogarithmicInfo(DistributionInfo[LogarithmicNP, LogarithmicEP, NumpyRealArray]):
    @override
    def nat_to_scipy_distribution(self, q: LogarithmicNP) -> Any:
        return ss.logser(np.exp(q.log_probability))

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> LogarithmicEP:
        return LogarithmicEP(jnp.asarray(rng.exponential(size=shape) + 1.0))


class NormalInfo(DistributionInfo[NormalNP, NormalEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: NormalEP) -> Any:
        return ss.norm(p.mean, np.sqrt(p.variance()))

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> NormalEP:
        mean = rng.normal(scale=4.0, size=shape)
        variance = rng.exponential(size=shape)
        return NormalEP(jnp.asarray(mean), jnp.asarray(mean ** 2 + variance))


class MultivariateFixedVarianceNormalInfo(DistributionInfo[MultivariateFixedVarianceNormalNP,
                                                           MultivariateFixedVarianceNormalEP,
                                                           NumpyRealArray]):
    @override
    def __init__(self, dimensions: int, variance: float):
        super().__init__()
        self.dimensions = dimensions
        self.variance = variance

    @override
    def exp_to_scipy_distribution(self, p: MultivariateFixedVarianceNormalEP) -> Any:
        cov = np.tile(np.eye(p.dimensions()), (*p.shape, 1, 1))
        for i in np.ndindex(*p.shape):
            cov[i] *= p.variance[i]
        return ScipyMultivariateNormal.from_mc(mean=np.asarray(p.mean), cov=np.asarray(cov))

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape
                                ) -> MultivariateFixedVarianceNormalEP:
        variance = self.variance * jnp.ones(shape)
        return MultivariateFixedVarianceNormalEP(
            jnp.asarray(rng.normal(size=(*shape, self.dimensions))),
            variance=variance)


class MultivariateUnitNormalInfo(DistributionInfo[MultivariateUnitNormalNP,
                                                  MultivariateUnitNormalEP,
                                                  NumpyRealArray]):
    @override
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    @override
    def exp_to_scipy_distribution(self, p: MultivariateUnitNormalEP) -> Any:
        return ScipyMultivariateNormal.from_mc(mean=np.asarray(p.mean))

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> MultivariateUnitNormalEP:
        return MultivariateUnitNormalEP(jnp.asarray(rng.normal(size=(*shape, self.dimensions))))


class IsotropicNormalInfo(DistributionInfo[IsotropicNormalNP, IsotropicNormalEP, NumpyRealArray]):
    @override
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    @override
    def exp_to_scipy_distribution(self, p: IsotropicNormalEP) -> Any:
        v = p.variance()
        e = np.eye(self.dimensions)
        return ScipyMultivariateNormal.from_mc(mean=np.asarray(p.mean),
                                               cov=np.asarray(np.multiply.outer(v, e)))

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> IsotropicNormalEP:
        mean = rng.normal(size=(*shape, self.dimensions))
        total_variance = self.dimensions * rng.exponential(size=shape)
        return IsotropicNormalEP(jnp.asarray(mean), jnp.sum(np.square(mean)) + total_variance)


class MultivariateDiagonalNormalInfo(DistributionInfo[MultivariateDiagonalNormalNP,
                                                      MultivariateDiagonalNormalEP,
                                                      NumpyRealArray]):
    @override
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    @override
    def exp_to_scipy_distribution(self, p: MultivariateDiagonalNormalEP) -> Any:
        return ScipyMultivariateNormal.from_mc(mean=np.asarray(p.mean),
                                               cov=create_diagonal(np.asarray(p.variance())))

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> MultivariateDiagonalNormalEP:
        dist_shape = (*shape, self.dimensions)
        mean = rng.normal(size=dist_shape)
        variance = rng.exponential(size=dist_shape)
        return MultivariateDiagonalNormalEP(jnp.asarray(mean), jnp.square(mean) + variance)


class MultivariateNormalInfo(DistributionInfo[MultivariateUnitNormalNP, MultivariateNormalEP,
                                              NumpyRealArray]):
    @override
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    @override
    def exp_to_scipy_distribution(self, p: MultivariateNormalEP) -> Any:
        # Correct numerical errors introduced by various conversions.
        v = np.asarray(p.variance())
        v_transpose = v.swapaxes(-1, -2)
        covariance = vectorized_tril(v) + vectorized_triu(v_transpose, 1)
        return ScipyMultivariateNormal.from_mc(mean=np.asarray(p.mean), cov=covariance)

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> MultivariateNormalEP:
        covariance = vectorized_real_covariance(rng, shape, self.dimensions)
        mean = rng.normal(size=(*shape, self.dimensions))
        second_moment = covariance + mean[..., :, np.newaxis] * mean[..., np.newaxis, :]
        return MultivariateNormalEP(jnp.asarray(mean), jnp.asarray(second_moment))


class ComplexNormalInfo(DistributionInfo[ComplexNormalNP, ComplexNormalEP, NumpyComplexArray]):
    @override
    def exp_to_scipy_distribution(self, p: ComplexNormalEP) -> Any:
        mean = np.asarray(p.mean)
        second_moment = np.asarray(p.second_moment)
        pseudo_second_moment = np.asarray(p.pseudo_second_moment)
        return ScipyComplexNormal(mean,
                                  second_moment - np_abs_square(mean),
                                  pseudo_second_moment - np.square(mean))

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> ComplexNormalEP:
        mean = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        variance = rng.exponential(size=shape)
        mean_j = jnp.asarray(mean)
        second_moment = jnp.asarray(np_abs_square(mean) + variance)
        pseudo_variance = (variance * rng.beta(2, 2, size=shape)
                           * np.exp(1j * rng.uniform(0, 2 * np.pi, size=shape)))
        pseudo_second_moment = jnp.asarray(np.square(mean) + pseudo_variance)
        return ComplexNormalEP(mean_j, second_moment, pseudo_second_moment)


class ComplexMultivariateUnitNormalInfo(DistributionInfo[ComplexMultivariateUnitNormalNP,
                                                         ComplexMultivariateUnitNormalEP,
                                                         NumpyComplexArray]):
    @override
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    @override
    def exp_to_scipy_distribution(self, p: ComplexMultivariateUnitNormalEP) -> Any:
        return ScipyComplexMultivariateNormal(mean=np.asarray(p.mean))

    @override
    def exp_parameter_generator(self,
                                rng: Generator,
                                shape: Shape) -> ComplexMultivariateUnitNormalEP:
        a = rng.normal(size=(*shape, self.dimensions))
        b = rng.normal(size=(*shape, self.dimensions))
        return ComplexMultivariateUnitNormalEP(jnp.asarray(a + 1j * b))


class ComplexCircularlySymmetricNormalInfo(DistributionInfo[ComplexCircularlySymmetricNormalNP,
                                                            ComplexCircularlySymmetricNormalEP,
                                                            NumpyComplexArray]):
    @override
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    @override
    def exp_to_scipy_distribution(self, p: ComplexCircularlySymmetricNormalEP) -> Any:
        return ScipyComplexMultivariateNormal(variance=np.asarray(p.variance))

    @override
    def exp_parameter_generator(self,
                                rng: Generator,
                                shape: Shape) -> ComplexCircularlySymmetricNormalEP:
        cc = vectorized_complex_covariance(rng, shape, self.dimensions)
        return ComplexCircularlySymmetricNormalEP(jnp.asarray(cc))


class ExponentialInfo(DistributionInfo[ExponentialNP, ExponentialEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: ExponentialEP) -> Any:
        return ss.expon(0, p.mean)

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> ExponentialEP:
        return ExponentialEP(jnp.asarray(rng.exponential(size=shape)))


class RayleighInfo(DistributionInfo[RayleighNP, RayleighEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: RayleighEP) -> Any:
        return ss.rayleigh(scale=np.sqrt(p.chi / 2.0))

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> RayleighEP:
        return RayleighEP(jnp.asarray(rng.exponential(size=shape)))


class BetaInfo(DistributionInfo[BetaNP, BetaEP, NumpyRealArray]):
    @override
    def nat_to_scipy_distribution(self, q: BetaNP) -> Any:
        n1 = q.alpha_minus_one + 1.0
        return ss.beta(n1[..., 0], n1[..., 1])

    @override
    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> BetaNP:
        return BetaNP(jnp.asarray(dirichlet_parameter_generator(2, rng, shape)))


class GammaInfo(DistributionInfo[GammaNP, GammaEP, NumpyRealArray]):
    @override
    def nat_to_scipy_distribution(self, q: GammaNP) -> Any:
        shape = q.shape_minus_one + 1.0
        scale = -1.0 / q.negative_rate
        return ss.gamma(shape, scale=scale)

    @override
    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> GammaNP:
        gamma_shape = jnp.asarray(rng.exponential(size=shape))
        rate = jnp.asarray(rng.exponential(size=shape))
        return GammaNP(-rate, gamma_shape - 1.0)


class DirichletInfo(DistributionInfo[DirichletNP, DirichletEP, NumpyRealArray]):
    @override
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    @override
    def nat_to_scipy_distribution(self, q: DirichletNP) -> Any:
        return ScipyDirichlet(np.asarray(q.alpha_minus_one) + 1.0)

    @override
    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> DirichletNP:
        return DirichletNP(jnp.asarray(dirichlet_parameter_generator(self.dimensions, rng, shape)))

    @override
    def scipy_to_exp_family_observation(self, x: NumpyRealArray) -> NumpyRealArray:
        return x[..., : -1]


class GeneralizedDirichletInfo(DistributionInfo[GeneralizedDirichletNP, GeneralizedDirichletEP,
                                                NumpyRealArray]):
    @override
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    @override
    def nat_to_scipy_distribution(self, q: GeneralizedDirichletNP) -> Any:
        alpha, beta = q.alpha_beta()
        return ScipyGeneralizedDirichlet(np.asarray(alpha), np.asarray(beta))

    @override
    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> GeneralizedDirichletNP:
        alpha_minus_one = dirichlet_parameter_generator(self.dimensions, rng, shape)
        gamma = dirichlet_parameter_generator(self.dimensions, rng, shape) + 1.0
        return GeneralizedDirichletNP(jnp.asarray(alpha_minus_one), jnp.asarray(gamma))


class VonMisesFisherInfo(DistributionInfo[VonMisesFisherNP, VonMisesFisherEP, NumpyRealArray]):
    @override
    def nat_to_scipy_distribution(self, q: VonMisesFisherNP) -> Any:
        kappa, angle = q.to_kappa_angle()
        return ScipyVonMises(np.asarray(kappa), np.asarray(angle))

    @override
    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> VonMisesFisherNP:
        return VonMisesFisherNP(jnp.asarray(rng.normal(size=(*shape, 2), scale=4.0)))

    @override
    def scipy_to_exp_family_observation(self, x: NumpyRealArray) -> NumpyRealArray:
        result = np.empty((*x.shape, 2))
        result[..., 0] = np.cos(x)
        result[..., 1] = np.sin(x)
        return result


class ChiSquareInfo(DistributionInfo[ChiSquareNP, ChiSquareEP, NumpyRealArray]):
    @override
    def nat_to_scipy_distribution(self, q: ChiSquareNP) -> Any:
        return ss.chi2((q.k_over_two_minus_one + 1.0) * 2.0)

    @override
    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> ChiSquareNP:
        return ChiSquareNP(jnp.asarray(rng.exponential(size=shape)))


class ChiInfo(DistributionInfo[ChiNP, ChiEP, NumpyRealArray]):
    @override
    def nat_to_scipy_distribution(self, q: ChiNP) -> Any:
        return ss.chi((q.k_over_two_minus_one + 1.0) * 2.0)

    @override
    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> ChiNP:
        return ChiNP(jnp.asarray(rng.exponential(size=shape)))


class WeibullInfo(DistributionInfo[WeibullNP, WeibullEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: WeibullEP) -> Any:
        scale = p.chi ** (1.0 / p.concentration)
        return ss.weibull_min(p.concentration, scale=scale)

    @override
    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> WeibullNP:
        equal_fixed_parameters = True
        concentration = (np.broadcast_to(rng.exponential(), shape)
                         if equal_fixed_parameters
                         else rng.exponential(size=shape)) + 1.0
        return WeibullNP(jnp.asarray(concentration),
                         jnp.asarray(-rng.exponential(size=shape) - 1.0))


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
