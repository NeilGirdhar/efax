"""
These tests ensure that our distributions match scipy's.
"""
import numpy as np
from numpy.testing import assert_allclose

from ..logarithmic import Logarithmic


def test_entropy(generator, distribution_info):
    """
    Test that the entropy calculation matches scipy's.
    """
    if isinstance(distribution_info.my_distribution,
                  (Logarithmic,)):
        return
    nat_parameters = distribution_info.nat_parameter_generator(
        generator, shape=(3, 2))
    my_distribution = distribution_info.my_distribution
    scipy_distribution = distribution_info.nat_to_scipy_distribution(
        nat_parameters)
    try:
        my_entropy = my_distribution.entropy(nat_parameters)
        scipy_entropy = scipy_distribution.entropy()
        if (scipy_entropy.shape
                and my_entropy.shape == scipy_entropy.shape[:-1]):
            scipy_entropy = scipy_entropy[..., 0]
        assert_allclose(my_entropy, scipy_entropy, rtol=1e-5)
    except NotImplementedError:
        pass


def test_pdf(generator, distribution_info):
    """
    Test that the density/mass function calculation matches scipy's.
    """
    if isinstance(distribution_info.my_distribution, Logarithmic):
        return
    for _ in range(10):
        nat_parameters = distribution_info.nat_parameter_generator(
            generator, shape=())
        my_distribution = distribution_info.my_distribution
        scipy_distribution = distribution_info.nat_to_scipy_distribution(
            nat_parameters)
        x = np.asarray(scipy_distribution.rvs(random_state=generator))
        my_x = np.asarray(distribution_info.my_observation(x))

        try:
            density = scipy_distribution.pdf(x)
        except AttributeError:
            density = scipy_distribution.pmf(x)
        except NotImplementedError:
            continue

        assert_allclose(
            my_distribution.pdf(nat_parameters, my_x),
            density,
            rtol=1e-4)


def test_maximum_likelihood_estimation(generator, distribution_info):
    """
    Test that maximum likelihood estimation from scipy-generated variates
    produce the same distribution from which they were drawn.
    """
    if isinstance(distribution_info.my_distribution, Logarithmic):
        return
    parameters = distribution_info.exp_parameter_generator(
        generator, shape=())
    my_distribution = distribution_info.my_distribution
    scipy_distribution = distribution_info.exp_to_scipy_distribution(
        parameters)
    x = scipy_distribution.rvs(random_state=generator, size=70000)
    my_x = distribution_info.my_observation(x)
    sufficient_stats = my_distribution.sufficient_statistics(my_x)
    shrink = int(my_distribution.num_parameters > 1)

    calculated_parameters = np.mean(
        sufficient_stats,
        axis=tuple(range(sufficient_stats.ndim - shrink)))

    assert_allclose(parameters, calculated_parameters,
                    rtol=2e-2)
