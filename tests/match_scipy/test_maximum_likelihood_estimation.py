from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
from jax import Array
from numpy.random import Generator
from tjax import assert_tree_allclose

from efax import (MaximumLikelihoodEstimator, NaturalParametrization, flat_dict_of_observations,
                  flat_dict_of_parameters, parameter_map, unflatten_mapping)

from ..create_info import (ComplexCircularlySymmetricNormalInfo, IsotropicNormalInfo,
                           MultivariateFixedVarianceNormalInfo, MultivariateNormalInfo,
                           VonMisesFisherInfo)
from ..distribution_info import DistributionInfo


def test_maximum_likelihood_estimation(
        generator: Generator,
        distribution_info: DistributionInfo[NaturalParametrization]
        ) -> None:
    """Test maximum likelihood estimation using SciPy.

    Test that maximum likelihood estimation from scipy-generated variates produce the same
    distribution from which they were drawn.
    """
    rtol = 2e-2
    if isinstance(distribution_info,
                  ComplexCircularlySymmetricNormalInfo | MultivariateNormalInfo
                  | VonMisesFisherInfo | MultivariateFixedVarianceNormalInfo):
        atol = 1e-2
    elif isinstance(distribution_info, IsotropicNormalInfo):
        atol = 1e-3
    else:
        atol = 1e-6
    n = 70000
    # Generate a distribution with expectation parameters.
    exp_parameters = distribution_info.exp_parameter_generator(generator, shape=())
    # Generate variates from the corresponding scipy distribution.
    scipy_distribution = distribution_info.exp_to_scipy_distribution(exp_parameters)
    scipy_x = scipy_distribution.sample(rng=generator, shape=(n,))
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
