from typing import Any, List

import numpy as np
from numpy.random import Generator
from scipy import stats as ss
from tjax import Shape

from efax import (BernoulliEP, BernoulliNP, BetaEP, BetaNP, ChiSquareEP, ChiSquareNP,
                  ComplexNormalEP, ComplexNormalNP, DirichletEP, DirichletNP, ExponentialEP,
                  ExponentialNP, GammaEP, GammaNP, GeometricEP, GeometricNP, IsotropicNormalEP,
                  IsotropicNormalNP, LogarithmicEP, LogarithmicNP, MultivariateNormalEP,
                  MultivariateUnitNormalEP, MultivariateUnitNormalNP, NegativeBinomialEP,
                  NegativeBinomialNP, NormalEP, NormalNP, PoissonEP, PoissonNP, ScipyComplexNormal,
                  ScipyDirichlet, VonMisesFisherEP, VonMisesFisherNP)

from .distribution_info import DistributionInfo


def dirichlet_parameter_generator(n: int, rng: Generator, shape: Shape) -> np.ndarray:
    # q can be as low as -1, but we prevent low values
    return rng.exponential(size=(*shape, n), scale=4.0) + 0.7


class BernoulliInfo(DistributionInfo[BernoulliNP, BernoulliEP]):
    def exp_to_scipy_distribution(self, p: BernoulliEP) -> Any:
        return ss.bernoulli(p.probability)

    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> BernoulliEP:
        return BernoulliEP(rng.uniform(size=shape))


class GeometricInfo(DistributionInfo[GeometricNP, GeometricEP]):
    def exp_to_scipy_distribution(self, p: GeometricEP) -> Any:
        # p is inverse odds
        return ss.geom(1.0 / (1.0 + p.mean))

    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> GeometricEP:
        return GeometricEP(rng.exponential(size=shape))

    def scipy_to_exp_family_observation(self, x: np.ndarray) -> np.ndarray:
        return x - 1


class PoissonInfo(DistributionInfo[PoissonNP, PoissonEP]):
    def exp_to_scipy_distribution(self, p: PoissonEP) -> Any:
        return ss.poisson(p.mean)

    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> PoissonEP:
        return PoissonEP(rng.exponential(size=shape))


class NegativeBinomialInfo(DistributionInfo[NegativeBinomialNP, NegativeBinomialEP]):
    def __init__(self, r: int):
        self.r = r

    def exp_to_scipy_distribution(self, p: NegativeBinomialEP) -> Any:
        return ss.nbinom(self.r, 1.0 / (1.0 + p.mean / p.failures))

    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> NegativeBinomialEP:
        return NegativeBinomialEP(self.r, rng.exponential(size=shape))


class LogarithmicInfo(DistributionInfo[LogarithmicNP, LogarithmicEP]):
    def nat_to_scipy_distribution(self, q: LogarithmicNP) -> Any:
        return ss.logser(np.exp(q.log_probability))

    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> LogarithmicEP:
        return LogarithmicEP(rng.exponential(size=shape) + 1.0)


class NormalInfo(DistributionInfo[NormalNP, NormalEP]):
    def exp_to_scipy_distribution(self, p: NormalEP) -> Any:
        return ss.norm(p.mean, np.sqrt(p.variance()))

    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> NormalEP:
        mean = rng.normal(scale=4.0, size=shape)
        variance = rng.exponential(size=shape)
        return NormalEP(mean, mean ** 2 + variance)


class MultivariateUnitNormalInfo(DistributionInfo[MultivariateUnitNormalNP,
                                                  MultivariateUnitNormalEP]):
    def __init__(self, num_parameters: int):
        self.num_parameters = num_parameters

    def exp_to_scipy_distribution(self, p: MultivariateUnitNormalEP) -> Any:
        return ss.multivariate_normal(mean=p.mean)

    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> MultivariateUnitNormalEP:
        if shape != ():
            raise ValueError
        return MultivariateUnitNormalEP(rng.normal(size=(*shape, self.num_parameters)))

    def supports_shape(self) -> bool:
        return False


class IsotropicNormalInfo(DistributionInfo[IsotropicNormalNP, IsotropicNormalEP]):
    def __init__(self, num_parameters: int):
        self.num_parameters = num_parameters

    def exp_to_scipy_distribution(self, p: IsotropicNormalEP) -> Any:
        return ss.multivariate_normal(mean=p.mean, cov=np.eye(self.num_parameters) * p.variance())

    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> IsotropicNormalEP:
        if shape != ():
            raise ValueError
        mean = rng.normal(size=(*shape, self.num_parameters))
        total_variance = self.num_parameters * rng.exponential(size=shape)
        return IsotropicNormalEP(mean, np.sum(np.square(mean)) + total_variance)

    def supports_shape(self) -> bool:
        return False


class MultivariateNormalInfo(DistributionInfo[MultivariateUnitNormalNP, MultivariateNormalEP]):
    def __init__(self, num_parameters: int):
        self.num_parameters = num_parameters

    def exp_to_scipy_distribution(self, p: MultivariateNormalEP) -> Any:
        # Correct numerical errors introduced by various conversions.
        covariance = np.tril(p.variance()) + np.triu(p.variance().T, 1)
        return ss.multivariate_normal(mean=p.mean, cov=covariance)

    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> MultivariateNormalEP:
        if shape != ():
            raise ValueError
        if self.num_parameters == 1:
            covariance = rng.exponential()
        else:
            eigenvalues = rng.exponential(size=self.num_parameters) + 1.0
            eigenvalues /= np.mean(eigenvalues)
            covariance = ss.random_correlation.rvs(eigenvalues, random_state=rng)
        mean = rng.normal(size=(*shape, self.num_parameters))
        second_moment = covariance + np.multiply.outer(mean, mean)
        return MultivariateNormalEP(mean, second_moment)

    def supports_shape(self) -> bool:
        return False


class ComplexNormalInfo(DistributionInfo[ComplexNormalNP, ComplexNormalEP]):
    def exp_to_scipy_distribution(self, p: ComplexNormalEP) -> Any:
        return ScipyComplexNormal(p.mean,
                                  (p.second_moment - p.mean.conjugate() * p.mean).real,
                                  p.pseudo_second_moment - np.square(p.mean))

    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> ComplexNormalEP:
        mean = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        variance = rng.exponential(size=shape)
        second_moment = mean.conjugate() * mean + variance
        pseudo_variance = (variance * rng.beta(2, 2, size=shape)
                           * np.exp(1j * rng.uniform(0, 2 * np.pi, size=shape)))
        pseudo_second_moment = np.square(mean) + pseudo_variance
        return ComplexNormalEP(mean, second_moment, pseudo_second_moment)


class ExponentialInfo(DistributionInfo[ExponentialNP, ExponentialEP]):
    def exp_to_scipy_distribution(self, p: ExponentialEP) -> Any:
        return ss.expon(0, p.mean)

    def exp_parameter_generator(self, rng: Generator, shape: Shape) -> ExponentialEP:
        return ExponentialEP(rng.exponential(size=shape))


class BetaInfo(DistributionInfo[BetaNP, BetaEP]):
    def nat_to_scipy_distribution(self, q: BetaNP) -> Any:
        n1 = q.alpha_minus_one + 1.0
        return ss.beta(n1[..., 0], n1[..., 1])

    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> BetaNP:
        return BetaNP(dirichlet_parameter_generator(2, rng, shape))


class GammaInfo(DistributionInfo[GammaNP, GammaEP]):
    def exp_to_scipy_distribution(self, p: GammaEP) -> Any:
        shape, scale = p.solve_for_shape_and_scale()
        return ss.gamma(shape, scale=scale)

    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> GammaNP:
        gamma_shape = rng.exponential(size=shape)
        rate = rng.exponential(size=shape)
        return GammaNP(-rate, gamma_shape - 1.0)


class DirichletInfo(DistributionInfo[DirichletNP, DirichletEP]):
    def __init__(self, num_parameters: int):
        self.num_parameters = num_parameters

    def nat_to_scipy_distribution(self, q: DirichletNP) -> Any:
        return ScipyDirichlet(q.alpha_minus_one + 1.0)

    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> DirichletNP:
        return DirichletNP(dirichlet_parameter_generator(self.num_parameters, rng, shape))

    def scipy_to_exp_family_observation(self, x: np.ndarray) -> np.ndarray:
        return x[..., : -1]


class VonMisesFisherInfo(DistributionInfo[VonMisesFisherNP, VonMisesFisherEP]):
    def nat_to_scipy_distribution(self, q: VonMisesFisherNP) -> Any:
        return ss.vonmises(*q.to_kappa_angle())

    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> VonMisesFisherNP:
        if shape != ():
            raise ValueError
        return VonMisesFisherNP(rng.normal(size=(*shape, 2), scale=4.0))

    def scipy_to_exp_family_observation(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        result = np.empty(x.shape + (2,))
        result[..., 0] = np.cos(x)
        result[..., 1] = np.sin(x)
        return result

    def supports_shape(self) -> bool:
        return False


class ChiSquareInfo(DistributionInfo[ChiSquareNP, ChiSquareEP]):
    def nat_to_scipy_distribution(self, q: ChiSquareNP) -> Any:
        return ss.chi2((q.k_over_two_minus_one + 1.0) * 2.0)

    def nat_parameter_generator(self, rng: Generator, shape: Shape) -> ChiSquareNP:
        return ChiSquareNP(rng.exponential(size=shape))


def create_infos() -> List[DistributionInfo[Any, Any]]:
    # Discrete
    bernoulli = BernoulliInfo()
    geometric = GeometricInfo()
    poisson = PoissonInfo()
    negative_binomial = NegativeBinomialInfo(3)
    logarithmic = LogarithmicInfo()

    # Continuous
    normal = NormalInfo()
    isotropic_normal = IsotropicNormalInfo(num_parameters=4)
    multivariate_normal = MultivariateNormalInfo(num_parameters=4)
    multivariate_unit_normal = MultivariateUnitNormalInfo(num_parameters=5)
    complex_normal = ComplexNormalInfo()
    exponential = ExponentialInfo()
    gamma = GammaInfo()
    beta = BetaInfo()
    dirichlet = DirichletInfo(5)
    von_mises = VonMisesFisherInfo()
    chi_square = ChiSquareInfo()

    return [bernoulli, beta, chi_square, complex_normal, dirichlet, exponential, gamma, geometric,
            isotropic_normal, logarithmic, multivariate_normal, multivariate_unit_normal,
            negative_binomial, normal, poisson, von_mises]
