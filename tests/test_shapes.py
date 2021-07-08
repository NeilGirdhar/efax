from typing import Any

import numpy as np
from numpy.random import Generator

from .distribution_info import DistributionInfo


def test_shapes(generator: Generator, distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    """
    Test that the methods produce the correct shapes.
    """
    shape = (3, 4)

    p = distribution_info.exp_parameter_generator(generator, shape=shape)
    q = distribution_info.nat_parameter_generator(generator, shape=shape)
    scipy_dist = distribution_info.exp_to_scipy_distribution(p)
    scipy_x = scipy_dist.rvs()
    x = distribution_info.scipy_to_exp_family_observation(scipy_x)

    def check(z: Any) -> None:
        for _, xf, support in z.parameters_name_value_support():
            assert xf.shape[:len(xf.shape) - support.axes()] == shape

    assert p.shape == shape
    assert q.shape == shape

    check(p)
    check(q)

    assert q.log_normalizer().shape == shape
    try:
        assert p.cross_entropy(q).shape == shape
    except NotImplementedError:
        pass
    try:
        assert q.entropy().shape == shape
    except NotImplementedError:
        pass
    assert q.carrier_measure(x).shape == shape
    try:
        assert p.expected_carrier_measure().shape == shape
    except NotImplementedError:
        pass
    assert q.pdf(x).shape == shape


def test_types(distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    if isinstance(distribution_info.exp_parameter_generator(np.random.default_rng(), ()), tuple):
        raise TypeError("This should return a number or an ndarray")
