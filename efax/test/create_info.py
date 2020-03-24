from functools import partial

import numpy as np
import scipy.stats as ss

from ..bernoulli import Bernoulli
from ..complex_normal import ComplexNormal
from ..dirichlet import Beta, Dirichlet
from ..exponential import Exponential
from ..gamma import Gamma
from ..logarithmic import Logarithmic
from ..negative_binomial import Geometric, NegativeBinomial
from ..normal import Normal
from ..normal_unit_variance import NormalUnitVariance
from ..poisson import Poisson
from ..scipy_replacement import ScipyComplexNormal, ScipyDirichlet
from ..von_mises import VonMises
from .distribution_info import DistributionInfo


def create_infos():
    # Discrete
    bernoulli = DistributionInfo(
        Bernoulli(),
        exp_to_scipy_distribution=(
            lambda params: ss.bernoulli(params[..., 0])),
        exp_parameter_generator=(
            lambda rng, shape: rng.uniform(size=(*shape, 1))))

    geometric = DistributionInfo(
        Geometric(),
        exp_to_scipy_distribution=lambda inv_odds: ss.geom(
            1.0 / (1.0 + inv_odds)),
        exp_parameter_generator=(
            lambda rng, shape: rng.exponential(size=(*shape, 1))),
        my_observation=lambda x: x - 1)

    poisson = DistributionInfo(
        Poisson(),
        exp_to_scipy_distribution=lambda q: ss.poisson(q[..., 0]),
        exp_parameter_generator=(
            lambda rng, shape: rng.exponential(size=(*shape, 1))))

    nb_r = 3
    negative_binomial = DistributionInfo(
        NegativeBinomial(nb_r),
        exp_to_scipy_distribution=lambda inv_odds: ss.nbinom(
            nb_r, 1.0 / (1.0 + inv_odds / nb_r)),
        exp_parameter_generator=(
            lambda rng, shape: rng.exponential(size=(*shape, 1))))

    logarithmic = DistributionInfo(
        Logarithmic(),
        nat_to_scipy_distribution=lambda nat: ss.logser(np.exp(nat)),
        exp_parameter_generator=(
            lambda rng, shape: rng.exponential(size=(*shape, 1)) + 1.0))

    # Continuous
    def normal_scipy_dist(p):
        mean = p[..., 0]
        second_moment = p[..., 1]
        variance = second_moment - np.square(mean)
        return ss.norm(mean, np.sqrt(variance))

    def normal_rng(rng, shape):
        mean = rng.normal(scale=4.0, size=shape)
        variance = rng.exponential(size=shape)
        return np.stack([mean, mean ** 2 + variance], axis=-1)

    normal = DistributionInfo(
        Normal(),
        exp_to_scipy_distribution=normal_scipy_dist,
        exp_parameter_generator=normal_rng)

    unit_variance_n = 1
    normal_unit_variance = DistributionInfo(
        NormalUnitVariance(num_parameters=unit_variance_n),
        exp_to_scipy_distribution=lambda mean: ss.norm(mean, unit_variance_n),
        exp_parameter_generator=(
            lambda rng, shape: rng.normal(size=(*shape, unit_variance_n))))

    def cn_to_scipy(p):
        mean = np.asarray(p[..., 0])
        second_moment = np.asarray(p[..., 1])
        pseudo_second_moment = np.asarray(p[..., 2])
        return ScipyComplexNormal(
            mean,
            (second_moment - mean.conjugate() * mean).real,
            pseudo_second_moment - np.square(mean))

    def cn_parameters(rng, shape):
        mean = rng.normal(size=shape) + 1j * rng.normal()
        variance = rng.exponential(size=shape)
        second_moment = mean.conjugate() * mean + variance
        pseudo_variance = (
            variance
            * rng.beta(2, 2, size=shape)
            * np.exp(1j * rng.uniform(0, 2 * np.pi, size=shape)))
        pseudo_second_moment = np.square(mean) + pseudo_variance
        return np.stack([mean, second_moment, pseudo_second_moment], axis=-1)

    complex_normal = DistributionInfo(
        ComplexNormal(),
        exp_to_scipy_distribution=cn_to_scipy,
        exp_parameter_generator=cn_parameters)

    exponential = DistributionInfo(
        Exponential(),
        exp_to_scipy_distribution=lambda mean: ss.expon(0, mean),
        exp_parameter_generator=(
            lambda rng, shape: rng.exponential(size=(*shape, 1))))

    def create_scipy_gamma(parameters):
        mean = parameters[..., 0]
        mean_log = parameters[..., 1]
        shape, scale = Gamma.solve_for_shape_and_scale(mean, mean_log)
        return ss.gamma(shape, scale=scale)

    def random_gamma_nat_parms(rng, shape):
        gamma_shape = rng.exponential(size=shape)
        rate = rng.exponential(size=shape)
        return np.stack([-rate, gamma_shape - 1.0], axis=-1)

    gamma = DistributionInfo(
        Gamma(),
        exp_to_scipy_distribution=create_scipy_gamma,
        nat_parameter_generator=random_gamma_nat_parms)

    def dirichlet_parameter_generator(n, rng, shape):
        # q can be as low as -1, but we prevent low values
        return rng.exponential(size=(*shape, n), scale=4.0) + 0.5

    def beta_my_observation(x):
        x = np.asarray(x)
        result = np.empty(x.shape + (2,))
        result[..., 0] = x
        result[..., 1] = 1.0 - x
        return result

    def beta_nat_to_scipy_distribution(nat):
        n1 = nat + 1.0
        return ss.beta(n1[..., 0], n1[..., 1])

    beta = DistributionInfo(
        Beta(),
        nat_to_scipy_distribution=beta_nat_to_scipy_distribution,
        nat_parameter_generator=partial(
            dirichlet_parameter_generator, 2),
        my_observation=beta_my_observation)

    dirichlet_n = 5
    dirichlet = DistributionInfo(
        Dirichlet(dirichlet_n),
        nat_to_scipy_distribution=lambda nat: ScipyDirichlet(
            nat + 1.0),
        nat_parameter_generator=partial(
            dirichlet_parameter_generator, dirichlet_n))

    def von_mises_my_observation(x):
        x = np.asarray(x)
        result = np.empty(x.shape + (2,))
        result[..., 0] = np.cos(x)
        result[..., 1] = np.sin(x)
        return result
    von_mises = DistributionInfo(
        VonMises(),
        nat_to_scipy_distribution=lambda nat: ss.vonmises(
            *VonMises.nat_to_kappa_angle(nat)),
        nat_parameter_generator=lambda rng, shape: rng.normal(
            size=(*shape, 2), scale=4.0),
        my_observation=von_mises_my_observation)

    cpu_distributions_local = [bernoulli,
                               geometric,
                               poisson,
                               negative_binomial,
                               logarithmic,
                               normal,
                               normal_unit_variance,
                               complex_normal,
                               exponential,
                               gamma,
                               beta,
                               dirichlet,
                               von_mises,
                               ]

    return cpu_distributions_local
