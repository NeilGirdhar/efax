from typing import Any, List

import numpy as np
from numpy.random import Generator
from scipy import stats as ss
from tjax import Shape

from efax import (Bernoulli, Beta, ChiSquare, ComplexNormal, Dirichlet, Exponential, Gamma,
                  Geometric, Logarithmic, NegativeBinomial, Normal, NormalUnitVariance, Poisson,
                  ScipyComplexNormal, ScipyDirichlet, VonMises)

from .distribution_info import DistributionInfo


def create_infos() -> List[DistributionInfo]:
    # Discrete
    class BernoulliInfo(DistributionInfo[Bernoulli]):
        def exp_to_scipy_distribution(self, p: np.ndarray) -> Any:
            return ss.bernoulli(p[..., 0])

        def exp_parameter_generator(self, rng: Generator, shape: Shape) -> np.ndarray:
            return rng.uniform(size=(*shape, 1))
    bernoulli = BernoulliInfo(Bernoulli())

    class GeometricInfo(DistributionInfo[Geometric]):
        def exp_to_scipy_distribution(self, p: np.ndarray) -> Any:
            # p is inverse odds
            return ss.geom(1.0 / (1.0 + p))

        def exp_parameter_generator(self, rng: Generator, shape: Shape) -> np.ndarray:
            return rng.exponential(size=(*shape, 1))

        def scipy_to_exp_family_observation(self, x: np.ndarray) -> np.ndarray:
            return x - 1
    geometric = GeometricInfo(Geometric())

    class PoissonInfo(DistributionInfo[Poisson]):
        def exp_to_scipy_distribution(self, p: np.ndarray) -> Any:
            return ss.poisson(p[..., 0])

        def exp_parameter_generator(self, rng: Generator, shape: Shape) -> np.ndarray:
            return rng.exponential(size=(*shape, 1))
    poisson = PoissonInfo(Poisson())

    class NegativeBinomialInfo(DistributionInfo[NegativeBinomial]):
        def exp_to_scipy_distribution(self, p: np.ndarray) -> Any:
            return ss.nbinom(self.exp_family.r, 1.0 / (1.0 + p / self.exp_family.r))

        def exp_parameter_generator(self, rng: Generator, shape: Shape) -> np.ndarray:
            return rng.exponential(size=(*shape, 1))
    negative_binomial = NegativeBinomialInfo(NegativeBinomial(3))

    class LogarithmicInfo(DistributionInfo[Logarithmic]):
        def nat_to_scipy_distribution(self, q: np.ndarray) -> Any:
            return ss.logser(np.exp(q))

        def exp_parameter_generator(self, rng: Generator, shape: Shape) -> np.ndarray:
            return rng.exponential(size=(*shape, 1)) + 1.0
    logarithmic = LogarithmicInfo(Logarithmic())

    # Continuous
    class NormalInfo(DistributionInfo[Normal]):
        def exp_to_scipy_distribution(self, p: np.ndarray) -> Any:
            mean = p[..., 0]
            second_moment = p[..., 1]
            variance = second_moment - np.square(mean)
            return ss.norm(mean, np.sqrt(variance))

        def exp_parameter_generator(self, rng: Generator, shape: Shape) -> np.ndarray:
            mean = rng.normal(scale=4.0, size=shape)
            variance = rng.exponential(size=shape)
            return np.stack([mean, mean ** 2 + variance], axis=-1)
    normal = NormalInfo(Normal())

    class NormalUnitVarianceInfo(DistributionInfo[NormalUnitVariance]):
        def exp_to_scipy_distribution(self, p: np.ndarray) -> Any:
            return ss.norm(p, self.exp_family.num_parameters)

        def exp_parameter_generator(self, rng: Generator, shape: Shape) -> np.ndarray:
            return rng.normal(size=(*shape, self.exp_family.num_parameters))
    normal_unit_variance = NormalUnitVarianceInfo(NormalUnitVariance(num_parameters=1))

    class ComplexNormalInfo(DistributionInfo[ComplexNormal]):
        def exp_to_scipy_distribution(self, p: np.ndarray) -> Any:
            mean = np.asarray(p[..., 0])
            second_moment = np.asarray(p[..., 1])
            pseudo_second_moment = np.asarray(p[..., 2])
            return ScipyComplexNormal(mean,
                                      (second_moment - mean.conjugate() * mean).real,
                                      pseudo_second_moment - np.square(mean))

        def exp_parameter_generator(self, rng: Generator, shape: Shape) -> np.ndarray:
            mean = rng.normal(size=shape) + 1j * rng.normal()
            variance = rng.exponential(size=shape)
            second_moment = mean.conjugate() * mean + variance
            pseudo_variance = (variance * rng.beta(2, 2, size=shape)
                               * np.exp(1j * rng.uniform(0, 2 * np.pi, size=shape)))
            pseudo_second_moment = np.square(mean) + pseudo_variance
            return np.stack([mean, second_moment, pseudo_second_moment], axis=-1)
    complex_normal = ComplexNormalInfo(ComplexNormal())

    class ExponentialInfo(DistributionInfo[Exponential]):
        def exp_to_scipy_distribution(self, p: np.ndarray) -> Any:
            return ss.expon(0, p)

        def exp_parameter_generator(self, rng: Generator, shape: Shape) -> np.ndarray:
            return rng.exponential(size=(*shape, 1))
    exponential = ExponentialInfo(Exponential())

    class GammaInfo(DistributionInfo[Gamma]):
        def exp_to_scipy_distribution(self, p: np.ndarray) -> Any:
            mean = p[..., 0]
            mean_log = p[..., 1]
            shape, scale = Gamma.solve_for_shape_and_scale(mean, mean_log)
            return ss.gamma(shape, scale=scale)

        def nat_parameter_generator(self, rng: Generator, shape: Shape) -> np.ndarray:
            gamma_shape = rng.exponential(size=shape)
            rate = rng.exponential(size=shape)
            return np.stack([-rate, gamma_shape - 1.0], axis=-1)
    gamma = GammaInfo(Gamma())

    def dirichlet_parameter_generator(n: int, rng: Generator, shape: Shape) -> np.ndarray:
        # q can be as low as -1, but we prevent low values
        return rng.exponential(size=(*shape, n), scale=4.0) + 0.7

    class BetaInfo(DistributionInfo[Beta]):
        def nat_to_scipy_distribution(self, q: np.ndarray) -> Any:
            n1 = q + 1.0
            return ss.beta(n1[..., 0], n1[..., 1])

        def nat_parameter_generator(self, rng: Generator, shape: Shape) -> np.ndarray:
            return dirichlet_parameter_generator(2, rng, shape)

        def scipy_to_exp_family_observation(self, x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            result = np.empty(x.shape + (2,))
            result[..., 0] = x
            result[..., 1] = 1.0 - x
            return result
    beta = BetaInfo(Beta())

    class DirichletInfo(DistributionInfo[Dirichlet]):
        def nat_to_scipy_distribution(self, q: np.ndarray) -> Any:
            return ScipyDirichlet(q + 1.0)

        def nat_parameter_generator(self, rng: Generator, shape: Shape) -> np.ndarray:
            return dirichlet_parameter_generator(self.exp_family.num_parameters, rng, shape)
    dirichlet = DirichletInfo(Dirichlet(5))

    class VonMisesInfo(DistributionInfo[VonMises]):
        def nat_to_scipy_distribution(self, q: np.ndarray) -> Any:
            return ss.vonmises(*VonMises.nat_to_kappa_angle(q))

        def nat_parameter_generator(self, rng: Generator, shape: Shape) -> np.ndarray:
            return rng.normal(size=(*shape, 2), scale=4.0)

        def scipy_to_exp_family_observation(self, x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            result = np.empty(x.shape + (2,))
            result[..., 0] = np.cos(x)
            result[..., 1] = np.sin(x)
            return result
    von_mises = VonMisesInfo(VonMises())

    class ChiSquareInfo(DistributionInfo[ChiSquare]):
        def nat_to_scipy_distribution(self, q: np.ndarray) -> Any:
            return ss.chi2((q + 1.0) * 2.0)

        def nat_parameter_generator(self, rng: Generator, shape: Shape) -> np.ndarray:
            return rng.exponential(size=(*shape, 1))
    chi_square = ChiSquareInfo(ChiSquare())

    return [bernoulli, beta, chi_square, complex_normal, dirichlet, exponential, gamma, geometric,
            logarithmic, negative_binomial, normal, normal_unit_variance, poisson, von_mises]
