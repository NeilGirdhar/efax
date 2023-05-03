"""These tests apply to only samplable distributions."""
from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
from jax.random import KeyArray
from jax.tree_util import tree_map
from numpy.random import Generator
from tjax import assert_tree_allclose

from efax import Multidimensional, Samplable

from .create_info import (ComplexCircularlySymmetricNormalInfo, ComplexMultivariateUnitNormalInfo,
                          ComplexNormalInfo, GammaInfo, IsotropicNormalInfo,
                          MultivariateDiagonalNormalInfo, MultivariateFixedVarianceNormalInfo,
                          MultivariateNormalInfo, MultivariateUnitNormalInfo, PoissonInfo)
from .distribution_info import DistributionInfo


def test_maximum_likelihood_estimation(generator: Generator,
                                       key: KeyArray,
                                       sampling_distribution_info: DistributionInfo[Any, Any, Any],
                                       *,
                                       distribution_name: None | str,
                                       natural: bool) -> None:
    """Test that sampling is consistent with maximum likelihood estimation.

    This tests samples variates from either natural or expectation parametrizations.  Calculates the
    mean of the sufficient statistics, and verifies that it equals the expectation parameters.
    """
    sampling_distribution_info.skip_if_deselected(distribution_name)
    distribution_shape = (4,)
    sample_shape = (1024, 32)
    sample_axes = tuple(range(len(sample_shape)))
    atol = (3.0
            if isinstance(sampling_distribution_info, IsotropicNormalInfo)
            else 2e-1
            if isinstance(sampling_distribution_info, ComplexCircularlySymmetricNormalInfo)
            else 1e-1
            if isinstance(sampling_distribution_info, (ComplexNormalInfo,
                                                       ComplexMultivariateUnitNormalInfo,
                                                       MultivariateDiagonalNormalInfo,
                                                       MultivariateNormalInfo,
                                                       MultivariateFixedVarianceNormalInfo,
                                                       MultivariateUnitNormalInfo))
            else 1e-2
            if isinstance(sampling_distribution_info, PoissonInfo)
            else 1e-6)
    rtol = (5e-2
            if isinstance(sampling_distribution_info, GammaInfo)
            else 4e-2)

    if natural:
        nat_parameters = sampling_distribution_info.nat_parameter_generator(generator,
                                                                            distribution_shape)
        nat_cls = type(nat_parameters)
        assert isinstance(nat_parameters, Samplable)
        exp_parameters = nat_parameters.to_exp()  # type: ignore[attr-defined]
        samples = nat_parameters.sample(key, sample_shape)
        dimensions = (nat_parameters.dimensions() if isinstance(nat_parameters, Multidimensional)
                      else 0)
        fixed_parameters = nat_parameters.fixed_parameters()
        support = nat_parameters.domain_support()
    else:
        exp_parameters = sampling_distribution_info.exp_parameter_generator(generator,
                                                                            distribution_shape)
        assert isinstance(exp_parameters, Samplable)
        samples = exp_parameters.sample(key, sample_shape)
        dimensions = (exp_parameters.dimensions() if isinstance(exp_parameters, Multidimensional)
                      else 0)
        nat_cls = exp_parameters.natural_parametrization_cls()  # type: ignore[attr-defined]
        fixed_parameters = exp_parameters.fixed_parameters()
        support = exp_parameters.domain_support()

    # Verify that the samples have the right shape.
    ideal_shape = (*sample_shape, *distribution_shape, *support.shape(dimensions))
    assert samples.shape == ideal_shape

    # Verify the maximum likelihood estimate.
    fixed_parameters = {name: jnp.broadcast_to(value, (*sample_shape, *distribution_shape))
                        for name, value in fixed_parameters.items()}
    sampled_exp_parameters = nat_cls.sufficient_statistics(samples, **fixed_parameters)
    ml_exp_parameters = tree_map(partial(jnp.mean, axis=sample_axes), sampled_exp_parameters)
    assert_tree_allclose(ml_exp_parameters, exp_parameters, rtol=rtol, atol=atol)
