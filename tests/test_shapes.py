from typing import Any

import numpy as np
from numpy.random import Generator

from efax import NaturalParametrization, parameter_names_values_axes

from .create_info import VonMisesFisherInfo
from .distribution_info import DistributionInfo


def test_shapes(generator: Generator, distribution_info: DistributionInfo) -> None:
    """
    Test that the methods produce the correct shapes.
    """
    if isinstance(distribution_info, VonMisesFisherInfo):
        return
    shape = (3, 4) if distribution_info.supports_shape() else ()

    p = distribution_info.exp_parameter_generator(generator, shape=shape)
    q = distribution_info.nat_parameter_generator(generator, shape=shape)
    scipy_dist = distribution_info.exp_to_scipy_distribution(p)
    scipy_x = scipy_dist.rvs()
    x = distribution_info.scipy_to_exp_family_observation(scipy_x)

    def check(np: NaturalParametrization, z: Any) -> None:
        for _, xf, n_axes in parameter_names_values_axes(z):
            assert xf.shape[:len(xf.shape) - n_axes] == shape

    check(q, q)
    check(q, p)

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


def test_types(distribution_info: DistributionInfo) -> None:
    if isinstance(distribution_info.exp_parameter_generator(np.random.default_rng(), ()), tuple):
        raise TypeError("This should return a number or an ndarray")
