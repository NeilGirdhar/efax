"""
These tests ensure that our distributions match scipy's.
"""
from functools import partial
from typing import Any

import numpy as np
from jax.tree_util import tree_map
from numpy.random import Generator
from numpy.testing import assert_allclose
from tjax import assert_tree_allclose

from .create_info import (ComplexCircularlySymmetricNormalInfo, ComplexNormalInfo,
                          MultivariateDiagonalNormalInfo, MultivariateNormalInfo)
from .distribution_info import DistributionInfo


def test_entropy(generator: Generator, distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    """
    Test that the entropy calculation matches scipy's.
    """
    shape = (3, 2)
    nat_parameters = distribution_info.nat_parameter_generator(generator, shape=shape)
    scipy_distribution = distribution_info.nat_to_scipy_distribution(nat_parameters)
    rtol = (1e-3
            if isinstance(distribution_info, ComplexCircularlySymmetricNormalInfo)
            else 4e-4
            if isinstance(distribution_info, ComplexNormalInfo)
            else 2.0e-5)
    try:
        my_entropy = nat_parameters.entropy()
        scipy_entropy = scipy_distribution.entropy()
        assert_allclose(my_entropy, scipy_entropy, rtol=rtol)  # type: ignore
    except NotImplementedError:
        pass


def test_pdf(generator: Generator, distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    """
    Test that the density/mass function calculation matches scipy's.
    """
    for _ in range(10):
        nat_parameters = distribution_info.nat_parameter_generator(generator, shape=())
        scipy_distribution = distribution_info.nat_to_scipy_distribution(nat_parameters)
        x = np.asarray(scipy_distribution.rvs(random_state=generator))
        my_x = np.asarray(distribution_info.scipy_to_exp_family_observation(x))
        my_density = nat_parameters.pdf(my_x)

        try:
            density = scipy_distribution.pdf(x)
        except AttributeError:
            density = scipy_distribution.pmf(x)
        except NotImplementedError:
            continue

        if isinstance(distribution_info, MultivariateDiagonalNormalInfo):
            atol = 1e-5
            rtol = 3e-4
        else:
            atol = 1e-5
            rtol = 1e-4
        assert_allclose(my_density, density, rtol=rtol, atol=atol)  # type: ignore


def test_maximum_likelihood_estimation(generator: Generator,
                                       distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    """
    Test that maximum likelihood estimation from scipy-generated variates produce the same
    distribution from which they were drawn.
    """
    if isinstance(distribution_info, ComplexCircularlySymmetricNormalInfo):
        atol = 2e-2
        rtol = 1e-3
    elif isinstance(distribution_info, (ComplexNormalInfo, MultivariateNormalInfo)):
        atol = 1e-2
        rtol = 1e-3
    else:
        atol = 1e-4
        rtol = 2e-2
    exp_parameters = distribution_info.exp_parameter_generator(generator, shape=())
    nat_parameters = exp_parameters.to_nat()
    scipy_distribution = distribution_info.exp_to_scipy_distribution(exp_parameters)
    x = scipy_distribution.rvs(random_state=generator, size=70000)
    my_x = distribution_info.scipy_to_exp_family_observation(x)
    sufficient_stats = nat_parameters.sufficient_statistics(my_x)

    calculated_parameters = tree_map(partial(np.mean, axis=0), sufficient_stats)

    assert_tree_allclose(exp_parameters, calculated_parameters, rtol=rtol, atol=atol)
