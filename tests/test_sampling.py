"""
These tests apply to only samplable distributions.
"""
from functools import partial
from typing import Any

import jax.numpy as jnp
import pytest
from jax.tree_util import tree_map
from numpy.random import Generator as NumpyGenerator
from tjax import Generator as TjaxGenerator
from tjax import assert_jax_allclose

from efax import Samplable

from .create_info import (ComplexNormalInfo, GammaInfo, MultivariateNormalInfo,
                          MultivariateUnitNormalInfo)
from .distribution_info import DistributionInfo


@pytest.mark.parametrize('natural', [False, True])
def test_maximum_likelihood_estimation(generator: NumpyGenerator,
                                       rng: TjaxGenerator,
                                       distribution_info: DistributionInfo[Any, Any],
                                       natural: bool) -> None:
    """
    Test that maximum likelihood estimation from scipy-generated variates produce the same
    distribution from which they were drawn.
    """
    distribution_shape = (4,) if distribution_info.supports_shape() else ()
    sample_shape = (1000, 10)
    sample_axes = tuple(range(len(sample_shape)))
    atol = (1e-2
            if isinstance(distribution_info, (ComplexNormalInfo, MultivariateNormalInfo,
                                              MultivariateUnitNormalInfo))
            else 1e-6)
    rtol = (5e-2
            if isinstance(distribution_info, GammaInfo)
            else 3e-2)

    if natural:
        nat_parameters = distribution_info.nat_parameter_generator(generator, distribution_shape)
        if not isinstance(nat_parameters, Samplable):
            return
        exp_parameters = nat_parameters.to_exp()  # type: ignore
        samples = nat_parameters.sample(rng, sample_shape)
    else:
        exp_parameters = distribution_info.exp_parameter_generator(generator, distribution_shape)
        if not isinstance(exp_parameters, Samplable):
            return
        nat_parameters = exp_parameters.to_nat()  # type: ignore
        samples = exp_parameters.sample(rng, sample_shape)
    # assert samples.shape == sample_shape TODO
    sampled_exp_parameters = nat_parameters.sufficient_statistics(samples)
    ml_exp_parameters = tree_map(partial(jnp.mean, axis=sample_axes), sampled_exp_parameters)
    assert_jax_allclose(exp_parameters, ml_exp_parameters, rtol=rtol, atol=atol)
