import numpy as np
from numpy.random import Generator

from efax import VonMises

from .distribution_info import DistributionInfo


def test_shapes(generator: Generator, distribution_info: DistributionInfo) -> None:
    """
    Test that the methods produce the correct shapes.
    """
    if isinstance(distribution_info.exp_family, VonMises):
        return
    shape = (3, 4)
    exp_family = distribution_info.exp_family

    p = distribution_info.exp_parameter_generator(generator, shape=shape)
    q = distribution_info.nat_parameter_generator(generator, shape=shape)
    scipy_dist = distribution_info.exp_to_scipy_distribution(p)
    scipy_x = scipy_dist.rvs()
    x = distribution_info.scipy_to_exp_family_observation(scipy_x)

    assert x.shape == shape + exp_family.shape_including_observations()
    assert p.shape == shape + exp_family.shape_including_parameters()
    assert q.shape == shape + exp_family.shape_including_parameters()
    assert exp_family.log_normalizer(q).shape == shape + exp_family.shape
    assert exp_family.nat_to_exp(q).shape == shape + exp_family.shape_including_parameters()
    assert exp_family.exp_to_nat(p).shape == shape + exp_family.shape_including_parameters()
    assert (exp_family.sufficient_statistics(x).shape
            == shape + exp_family.shape_including_parameters())
    try:
        assert exp_family.cross_entropy(p, q).shape == shape + exp_family.shape
    except NotImplementedError:
        pass
    try:
        assert exp_family.entropy(q).shape == shape + exp_family.shape
    except NotImplementedError:
        pass
    assert exp_family.carrier_measure(x).shape == shape + exp_family.shape
    try:
        assert exp_family.expected_carrier_measure(p).shape == shape + exp_family.shape
    except NotImplementedError:
        pass
    assert exp_family.pdf(q, x).shape == shape + exp_family.shape


def test_types(distribution_info: DistributionInfo) -> None:
    if isinstance(distribution_info.exp_parameter_generator(np.random.default_rng(), ()), tuple):
        raise TypeError("This should return a number or an ndarray")
