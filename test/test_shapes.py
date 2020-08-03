import numpy as np
from numpy.random import Generator

from .distribution_info import DistributionInfo


def test_shapes(generator: Generator, distribution_info: DistributionInfo) -> None:
    """
    Test that the methods produce the correct shapes.
    """
    exp_family = distribution_info.exp_family

    ps = (3, 4)
    qs = (5, 2)

    p = distribution_info.exp_parameter_generator(generator, shape=ps)
    q = distribution_info.nat_parameter_generator(generator, shape=qs)

    assert p.shape == ps + exp_family.shape_including_parameters()
    assert q.shape == qs + exp_family.shape_including_parameters()
    assert exp_family.log_normalizer(q).shape == qs + exp_family.shape
    assert exp_family.nat_to_exp(q).shape == qs + exp_family.shape_including_parameters()
    assert exp_family.exp_to_nat(p).shape == ps + exp_family.shape_including_parameters()
    # assert exp_family.sufficient_statistics(x).shape == ps + exp_family.shape_including_parameters()


def test_types(distribution_info: DistributionInfo) -> None:
    if isinstance(distribution_info.exp_parameter_generator(np.random.default_rng(), ()), tuple):
        raise TypeError("This should return a number or an ndarray")

