"""
These tests ensure that our distributions match scipy's.
"""
from functools import partial

import numpy as np
from jax.tree_util import tree_map
from numpy.random import Generator
from numpy.testing import assert_allclose
from tjax import assert_jax_allclose

from .create_info import (ComplexNormalInfo, LogarithmicInfo, MultivariateNormalInfo,
                          VonMisesFisherInfo)
from .distribution_info import DistributionInfo


def test_entropy(generator, distribution_info):
    """
    Test that the entropy calculation matches scipy's.
    """
    if isinstance(distribution_info, (LogarithmicInfo, VonMisesFisherInfo)):
        return
    shape = (3, 2) if distribution_info.supports_shape() else ()
    nat_parameters = distribution_info.nat_parameter_generator(generator, shape=shape)
    scipy_distribution = distribution_info.nat_to_scipy_distribution(nat_parameters)
    rtol = 3e-4 if isinstance(distribution_info, ComplexNormalInfo) else 1.5e-5
    try:
        my_entropy = nat_parameters.entropy()
        scipy_entropy = scipy_distribution.entropy()
        assert_allclose(my_entropy, scipy_entropy, rtol=rtol)
    except NotImplementedError:
        pass


def test_pdf(generator: Generator, distribution_info: DistributionInfo):
    """
    Test that the density/mass function calculation matches scipy's.
    """
    if isinstance(distribution_info, (LogarithmicInfo, VonMisesFisherInfo)):
        return
    for _ in range(10):
        nat_parameters = distribution_info.nat_parameter_generator(generator, shape=())
        scipy_distribution = distribution_info.nat_to_scipy_distribution(nat_parameters)
        x = np.asarray(scipy_distribution.rvs(random_state=generator))
        my_x = np.asarray(distribution_info.scipy_to_exp_family_observation(x))

        try:
            density = scipy_distribution.pdf(x)
        except AttributeError:
            density = scipy_distribution.pmf(x)
        except NotImplementedError:
            continue

        assert_allclose(nat_parameters.pdf(my_x), density, rtol=1e-4)


def test_maximum_likelihood_estimation(generator, distribution_info):
    """
    Test that maximum likelihood estimation from scipy-generated variates produce the same
    distribution from which they were drawn.
    """
    if isinstance(distribution_info, LogarithmicInfo):
        return
    rtol = (8e-2
            if isinstance(distribution_info, (ComplexNormalInfo, MultivariateNormalInfo))
            else 2e-2)
    some_nat_parameters = distribution_info.nat_parameter_generator(generator, shape=())
    exp_parameters = distribution_info.exp_parameter_generator(generator, shape=())
    scipy_distribution = distribution_info.exp_to_scipy_distribution(exp_parameters)
    x = scipy_distribution.rvs(random_state=generator, size=70000)
    my_x = distribution_info.scipy_to_exp_family_observation(x)
    sufficient_stats = some_nat_parameters.sufficient_statistics(my_x)

    calculated_parameters = tree_map(partial(np.mean, axis=0), sufficient_stats)

    assert_jax_allclose(exp_parameters, calculated_parameters, rtol=rtol)
