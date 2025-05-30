from __future__ import annotations

from numpy.random import Generator
from numpy.testing import assert_allclose

from efax import HasEntropyEP, HasEntropyNP

from ..create_info import ChiInfo, ChiSquareInfo
from ..distribution_info import DistributionInfo


def test_nat_entropy(generator: Generator,
                     entropy_distribution_info: DistributionInfo
                     ) -> None:
    """Test that the entropy calculation matches scipy's."""
    shape = (7, 13)
    nat_parameters = entropy_distribution_info.nat_parameter_generator(generator, shape=shape)
    assert isinstance(nat_parameters, HasEntropyNP)
    scipy_distribution = entropy_distribution_info.nat_to_scipy_distribution(nat_parameters)
    rtol = 2e-5
    my_entropy = nat_parameters.entropy()
    scipy_entropy = scipy_distribution.entropy()
    assert_allclose(my_entropy, scipy_entropy, rtol=rtol)


def test_exp_entropy(generator: Generator,
                     entropy_distribution_info: DistributionInfo
                     ) -> None:
    """Test that the entropy calculation matches scipy's."""
    shape = (7, 13)
    exp_parameters = entropy_distribution_info.exp_parameter_generator(generator, shape=shape)
    assert isinstance(exp_parameters, HasEntropyEP)
    scipy_distribution = entropy_distribution_info.exp_to_scipy_distribution(exp_parameters)
    rtol = (1e-5
            if isinstance(entropy_distribution_info, ChiInfo | ChiSquareInfo)
            else 1e-6)
    my_entropy = exp_parameters.entropy()
    scipy_entropy = scipy_distribution.entropy()
    assert_allclose(my_entropy, scipy_entropy, rtol=rtol)
