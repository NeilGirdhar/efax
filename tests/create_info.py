from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import scipy.stats as ss
from jax.dtypes import canonicalize_dtype
from numpy.random import Generator
from tjax import NumpyComplexArray, NumpyRealArray, Shape, abs_square, create_diagonal_array
from typing_extensions import override

from efax import (BernoulliEP, BernoulliNP, BetaEP, BetaNP, ChiEP, ChiNP, ChiSquareEP, ChiSquareNP,
                  ComplexCircularlySymmetricNormalEP, ComplexCircularlySymmetricNormalNP,
                  ComplexMultivariateUnitNormalEP, ComplexMultivariateUnitNormalNP, ComplexNormalEP,
                  ComplexNormalNP, ComplexUnitNormalEP, ComplexUnitNormalNP, DirichletEP,
                  DirichletNP, ExponentialEP, ExponentialNP, GammaEP, GammaNP,
                  GeneralizedDirichletEP, GeneralizedDirichletNP, GeometricEP, GeometricNP,
                  IsotropicNormalEP, IsotropicNormalNP, LogarithmicEP, LogarithmicNP,
                  MultivariateDiagonalNormalEP, MultivariateDiagonalNormalNP,
                  MultivariateFixedVarianceNormalEP, MultivariateFixedVarianceNormalNP,
                  MultivariateNormalEP, MultivariateNormalNP, MultivariateUnitNormalEP,
                  MultivariateUnitNormalNP, NegativeBinomialEP, NegativeBinomialNP, NormalEP,
                  NormalNP, PoissonEP, PoissonNP, RayleighEP, RayleighNP,
                  ScipyComplexMultivariateNormal, ScipyComplexNormal, ScipyDirichlet,
                  ScipyGeneralizedDirichlet, ScipyMultivariateNormal, ScipyVonMises, UnitNormalEP,
                  UnitNormalNP, VonMisesFisherEP, VonMisesFisherNP, WeibullEP, WeibullNP)

from .distribution_info import DistributionInfo


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

    @override
    def exp_class(self) -> type[BernoulliEP]:
        return BernoulliEP

    @override
    def nat_class(self) -> type[BernoulliNP]:
        return BernoulliNP


class GeometricInfo(DistributionInfo[GeometricNP, GeometricEP, NumpyRealArray]):
    @override
    def exp_to_scipy_distribution(self, p: GeometricEP) -> Any:
        # Scipy uses a different definition geometric distribution.  The parameter pis inverse odds.
        return ss.geom(1.0 / (1.0 + p.mean))

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> GeometricEP:
        p = rng.uniform(size=shape)
        return GeometricEP(jnp.asarray(1.0 / p))

    @override
    def scipy_to_exp_family_observation(self, x: NumpyRealArray) -> NumpyRealArray:
        return x - 1

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
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> PoissonEP:
        return PoissonEP(jnp.asarray(rng.exponential(size=shape)))

    @override
    def exp_class(self) -> type[PoissonEP]:
        return PoissonEP

    @override
    def nat_class(self) -> type[PoissonNP]:
        return PoissonNP


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
                                  self.r * jnp.ones(shape, dtype=canonicalize_dtype(int)))

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
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> LogarithmicEP:
        return LogarithmicEP(jnp.asarray(rng.exponential(size=shape) + 1.0))

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
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> NormalEP:
        mean = rng.normal(scale=4.0, size=shape)
        variance = rng.exponential(size=shape)
        return NormalEP(jnp.asarray(mean), jnp.asarray(mean ** 2 + variance))

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
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> UnitNormalEP:
        mean = rng.normal(scale=4.0, size=shape)
        return UnitNormalEP(jnp.asarray(mean))

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
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    @override
    def exp_to_scipy_distribution(self, p: MultivariateUnitNormalEP) -> Any:
        return ScipyMultivariateNormal.from_mc(mean=np.asarray(p.mean))

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> MultivariateUnitNormalEP:
        return MultivariateUnitNormalEP(jnp.asarray(rng.normal(size=(*shape, self.dimensions))))

    @override
    def exp_class(self) -> type[MultivariateUnitNormalEP]:
        return MultivariateUnitNormalEP

    @override
    def nat_class(self) -> type[MultivariateUnitNormalNP]:
        return MultivariateUnitNormalNP


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
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    @override
    def exp_to_scipy_distribution(self, p: MultivariateDiagonalNormalEP) -> Any:
        variance = np.asarray(p.variance())
        covariance = create_diagonal_array(variance)
        return ScipyMultivariateNormal.from_mc(mean=np.asarray(p.mean), cov=covariance)

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> MultivariateDiagonalNormalEP:
        dist_shape = (*shape, self.dimensions)
        mean = rng.normal(size=dist_shape)
        variance = rng.exponential(size=dist_shape)
        return MultivariateDiagonalNormalEP(jnp.asarray(mean), jnp.square(mean) + variance)

    @override
    def exp_class(self) -> type[MultivariateDiagonalNormalEP]:
        return MultivariateDiagonalNormalEP

    @override
    def nat_class(self) -> type[MultivariateDiagonalNormalNP]:
        return MultivariateDiagonalNormalNP


class MultivariateNormalInfo(DistributionInfo[MultivariateNormalNP, MultivariateNormalEP,
                                              NumpyRealArray]):
    @override
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    @override
    def exp_to_scipy_distribution(self, p: MultivariateNormalEP) -> Any:
        # Correct numerical errors introduced by various conversions.
        mean = np.asarray(p.mean, dtype=np.float64)
        v = np.asarray(p.variance(), dtype=np.float64)
        v_transpose = v.swapaxes(-1, -2)
        covariance = np.tril(v) + np.triu(v_transpose, 1)
        return ScipyMultivariateNormal.from_mc(mean=mean, cov=covariance)

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> MultivariateNormalEP:
        covariance = vectorized_real_covariance(rng, shape, self.dimensions)
        mean = rng.normal(size=(*shape, self.dimensions))
        second_moment = covariance + mean[..., :, np.newaxis] * mean[..., np.newaxis, :]
        return MultivariateNormalEP(jnp.asarray(mean), jnp.asarray(second_moment))

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
        variance = np.ones_like(mean.real)
        pseudo_variance = np.zeros_like(mean)
        return ScipyComplexNormal(mean, variance, pseudo_variance)

    @override
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> ComplexUnitNormalEP:
        mean = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        mean_j = jnp.asarray(mean)
        return ComplexUnitNormalEP(mean_j)

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
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> ComplexNormalEP:
        mean = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        variance = rng.exponential(size=shape)
        mean_j = jnp.asarray(mean)
        second_moment = jnp.asarray(abs_square(mean) + variance)
        pseudo_variance = (variance * rng.beta(2, 2, size=shape)
                           * np.exp(1j * rng.uniform(0, 2 * np.pi, size=shape)))
        pseudo_second_moment = jnp.asarray(np.square(mean) + pseudo_variance)
        return ComplexNormalEP(mean_j, second_moment, pseudo_second_moment)

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
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> ExponentialEP:
        return ExponentialEP(jnp.asarray(rng.exponential(size=shape)))

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
    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> RayleighEP:
        return RayleighEP(jnp.asarray(rng.exponential(size=shape)))

    @override
    def exp_class(self) -> type[RayleighEP]:
        return RayleighEP

    @override
    def nat_class(self) -> type[RayleighNP]:
        return RayleighNP


class BetaInfo(DistributionInfo[BetaNP, BetaEP, NumpyRealArray]):
    @override
    def nat_to_scipy_distribution(self, q: BetaNP) -> Any:
        n1 = q.alpha_minus_one + 1.0
        return ss.beta(n1[..., 0], n1[..., 1])

    @override
    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> BetaNP:
        return BetaNP(jnp.asarray(dirichlet_parameter_generator(2, rng, shape)))

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
    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> GammaNP:
        gamma_shape = jnp.asarray(rng.exponential(size=shape))
        rate = jnp.asarray(rng.exponential(size=shape))
        return GammaNP(-rate, gamma_shape - 1.0)

    @override
    def exp_class(self) -> type[GammaEP]:
        return GammaEP

    @override
    def nat_class(self) -> type[GammaNP]:
        return GammaNP


class DirichletInfo(DistributionInfo[DirichletNP, DirichletEP, NumpyRealArray]):
    @override
    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    @override
    def nat_to_scipy_distribution(self, q: DirichletNP) -> Any:
        return ScipyDirichlet(np.asarray(q.alpha_minus_one, dtype=np.float64) + 1.0)

    @override
    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> DirichletNP:
        return DirichletNP(jnp.asarray(dirichlet_parameter_generator(self.dimensions, rng, shape)))

    @override
    def scipy_to_exp_family_observation(self, x: NumpyRealArray) -> NumpyRealArray:
        return x[..., : -1]

    @override
    def exp_class(self) -> type[DirichletEP]:
        return DirichletEP

    @override
    def nat_class(self) -> type[DirichletNP]:
        return DirichletNP


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

    @override
    def exp_class(self) -> type[GeneralizedDirichletEP]:
        return GeneralizedDirichletEP

    @override
    def nat_class(self) -> type[GeneralizedDirichletNP]:
        return GeneralizedDirichletNP


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
    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> ChiSquareNP:
        return ChiSquareNP(jnp.asarray(rng.exponential(size=shape)))

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
    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> ChiNP:
        return ChiNP(jnp.asarray(rng.exponential(size=shape)))

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
    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> WeibullNP:
        equal_fixed_parameters = True
        concentration = (np.broadcast_to(rng.exponential(), shape)
                         if equal_fixed_parameters
                         else rng.exponential(size=shape)) + 1.0
        return WeibullNP(jnp.asarray(concentration),
                         jnp.asarray(-rng.exponential(size=shape) - 1.0))

    @override
    def exp_class(self) -> type[WeibullEP]:
        return WeibullEP

    @override
    def nat_class(self) -> type[WeibullNP]:
        return WeibullNP


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
            DirichletInfo(5),
            ExponentialInfo(),
            GammaInfo(),
            GeneralizedDirichletInfo(5),
            GeometricInfo(),
            IsotropicNormalInfo(dimensions=4),
            LogarithmicInfo(),
            MultivariateDiagonalNormalInfo(dimensions=4),
            MultivariateFixedVarianceNormalInfo(dimensions=5, variance=3.0),
            MultivariateNormalInfo(dimensions=4),
            MultivariateUnitNormalInfo(dimensions=5),
            NegativeBinomialInfo(3),
            NormalInfo(),
            PoissonInfo(),
            RayleighInfo(),
            UnitNormalInfo(),
            VonMisesFisherInfo(),
            WeibullInfo(),
            ]
