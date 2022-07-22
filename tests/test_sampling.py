"""
These tests apply to only samplable distributions.
"""
from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
from jax.tree_util import tree_map
from numpy.random import Generator as NumpyGenerator
from tjax import Generator as TjaxGenerator
from tjax import assert_tree_allclose

from efax import Samplable

from .create_info import (ComplexCircularlySymmetricNormalInfo, ComplexMultivariateUnitNormalInfo,
                          ComplexNormalInfo, GammaInfo, IsotropicNormalInfo,
                          MultivariateDiagonalNormalInfo, MultivariateFixedVarianceNormalInfo,
                          MultivariateNormalInfo, MultivariateUnitNormalInfo, PoissonInfo)
from .distribution_info import DistributionInfo


def test_maximum_likelihood_estimation(generator: NumpyGenerator,
                                       rng: TjaxGenerator,
                                       sampling_distribution_info: DistributionInfo[Any, Any, Any],
                                       natural: bool) -> None:
    """
    Test that maximum likelihood estimation from scipy-generated variates produce the same
    distribution from which they were drawn.
    """
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
        assert isinstance(nat_parameters, Samplable)
        exp_parameters = nat_parameters.to_exp()  # type: ignore[attr-defined]
        samples = nat_parameters.sample(rng, sample_shape)
    else:
        exp_parameters = sampling_distribution_info.exp_parameter_generator(generator,
                                                                            distribution_shape)
        assert isinstance(exp_parameters, Samplable)
        nat_parameters = exp_parameters.to_nat()  # type: ignore[attr-defined]
        samples = exp_parameters.sample(rng, sample_shape)
    assert samples.shape[:len(sample_shape)] == sample_shape
    sampled_exp_parameters = nat_parameters.sufficient_statistics(samples)  # pyright: ignore
    ml_exp_parameters = tree_map(partial(jnp.mean, axis=sample_axes), sampled_exp_parameters)
    assert_tree_allclose(ml_exp_parameters, exp_parameters, rtol=rtol, atol=atol)
