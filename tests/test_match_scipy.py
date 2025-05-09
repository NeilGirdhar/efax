"""These tests ensure that our distributions match scipy's."""
from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
from jax import Array
from numpy.random import Generator
from numpy.testing import assert_allclose
from tjax import JaxComplexArray, assert_tree_allclose

from efax import (HasEntropyEP, HasEntropyNP, JointDistributionN, MaximumLikelihoodEstimator,
                  Multidimensional, NaturalParametrization, SimpleDistribution,
                  flat_dict_of_observations, flat_dict_of_parameters, parameter_map,
                  unflatten_mapping)

from .create_info import (BetaInfo, ComplexCircularlySymmetricNormalInfo, ComplexNormalInfo,
                          DirichletInfo, IsotropicNormalInfo, MultivariateDiagonalNormalInfo,
                          MultivariateFixedVarianceNormalInfo, MultivariateNormalInfo,
                          VonMisesFisherInfo)
from .distribution_info import DistributionInfo


def test_nat_entropy(generator: Generator,
                     entropy_distribution_info: DistributionInfo[Any, Any, Any]
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
                     entropy_distribution_info: DistributionInfo[Any, Any, Any]
                     ) -> None:
    """Test that the entropy calculation matches scipy's."""
    shape = (7, 13)
    exp_parameters = entropy_distribution_info.exp_parameter_generator(generator, shape=shape)
    assert isinstance(exp_parameters, HasEntropyEP)
    scipy_distribution = entropy_distribution_info.exp_to_scipy_distribution(exp_parameters)
    rtol = (8e-2
            if isinstance(entropy_distribution_info, BetaInfo)
            else 4e-2
            if isinstance(entropy_distribution_info, DirichletInfo)
            else 2e-5)
    my_entropy = exp_parameters.entropy()
    scipy_entropy = scipy_distribution.entropy()
    assert_allclose(my_entropy, scipy_entropy, rtol=rtol)


def check_observation_shape(nat_parameters: NaturalParametrization[Any, Any],
                            efax_x: JaxComplexArray | dict[str, Any],
                            distribution_shape: tuple[int, ...],
                            ) -> None:
    """Verify that the sufficient statistics have the right shape."""
    if isinstance(nat_parameters, JointDistributionN):
        assert isinstance(efax_x, dict)
        for name, value in nat_parameters.sub_distributions().items():
            check_observation_shape(value, efax_x[name], distribution_shape)
        return
    assert isinstance(nat_parameters, SimpleDistribution)  # type: ignore[unreachable]
    assert isinstance(efax_x, Array)  # type: ignore[unreachable]
    dimensions = (nat_parameters.dimensions()
                  if isinstance(nat_parameters, Multidimensional)
                  else 0)
    ideal_shape = distribution_shape + nat_parameters.domain_support().shape(dimensions)
    assert efax_x.shape == ideal_shape


def test_pdf(generator: Generator, distribution_info: DistributionInfo[Any, Any, Any]) -> None:
    """Test that the density/mass function calculation matches scipy's."""
    distribution_shape = (10,)
    nat_parameters = distribution_info.nat_parameter_generator(generator, shape=distribution_shape)
    scipy_distribution = distribution_info.nat_to_scipy_distribution(nat_parameters)
    scipy_x = scipy_distribution.rvs(random_state=generator)
    efax_x = distribution_info.scipy_to_exp_family_observation(scipy_x)
    check_observation_shape(nat_parameters, efax_x, distribution_shape)

    # Verify that the density matches scipy.
    efax_density = np.asarray(nat_parameters.pdf(efax_x), dtype=np.float64)
    try:
        scipy_density = scipy_distribution.pdf(scipy_x)
    except AttributeError:
        scipy_density = scipy_distribution.pmf(scipy_x)

    if isinstance(distribution_info, MultivariateDiagonalNormalInfo):
        atol = 1e-5
        rtol = 3e-4
    else:
        atol = 1e-5
        rtol = 1e-4
    assert_allclose(efax_density, scipy_density, rtol=rtol, atol=atol)


def test_maximum_likelihood_estimation(
        generator: Generator,
        distribution_info: DistributionInfo[NaturalParametrization[Any, Any], Any, Any]
        ) -> None:
    """Test maximum likelihood estimation using SciPy.

    Test that maximum likelihood estimation from scipy-generated variates produce the same
    distribution from which they were drawn.
    """
    if isinstance(distribution_info,
                  ComplexCircularlySymmetricNormalInfo | MultivariateNormalInfo
                  | VonMisesFisherInfo | MultivariateFixedVarianceNormalInfo):
        atol = 2e-2
        rtol = 2e-2
    elif isinstance(distribution_info, ComplexNormalInfo | IsotropicNormalInfo):
        atol = 1e-3
        rtol = 2e-2
    else:
        atol = 1e-4
        rtol = 2e-2
    n = 70000
    # Generate a distribution with expectation parameters.
    exp_parameters = distribution_info.exp_parameter_generator(generator, shape=())
    # Generate variates from the corresponding scipy distribution.
    scipy_distribution = distribution_info.exp_to_scipy_distribution(exp_parameters)
    scipy_x = scipy_distribution.rvs(random_state=generator, size=n)
    # Convert the variates to sufficient statistics.
    efax_x = distribution_info.scipy_to_exp_family_observation(scipy_x)
    flat_efax_x = flat_dict_of_observations(efax_x)
    flat_parameters = flat_dict_of_parameters(exp_parameters)
    flat_efax_x_clamped = {path: flat_parameters[path].domain_support().clamp(value)
                           for path, value in flat_efax_x.items()}
    efax_x_clamped: Array | dict[str, Any] = (flat_efax_x_clamped[()]
                                              if flat_efax_x_clamped.keys() == {()}
                                              else unflatten_mapping(flat_efax_x_clamped))
    estimator = MaximumLikelihoodEstimator.create_estimator(exp_parameters)
    sufficient_stats = estimator.sufficient_statistics(efax_x_clamped)

    # Verify that the mean of the sufficient statistics equals the expectation parameters.
    calculated_parameters = parameter_map(partial(np.mean, axis=0),  # type: ignore[arg-type]
                                          sufficient_stats)
    assert_tree_allclose(exp_parameters, calculated_parameters, rtol=rtol, atol=atol)
