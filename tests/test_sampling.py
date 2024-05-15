"""These tests apply to only samplable distributions."""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import hessian, jacrev, vmap
from jax.random import split
from numpy.random import Generator
from tjax import JaxArray, JaxRealArray, KeyArray, assert_tree_allclose

from efax import Multidimensional, Parametrization, Samplable, parameter_mean

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
            if isinstance(sampling_distribution_info,
                          ComplexNormalInfo | ComplexMultivariateUnitNormalInfo |
                          MultivariateDiagonalNormalInfo | MultivariateNormalInfo |
                          MultivariateFixedVarianceNormalInfo | MultivariateUnitNormalInfo)
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
    else:
        exp_parameters = sampling_distribution_info.exp_parameter_generator(generator,
                                                                            distribution_shape)
        assert isinstance(exp_parameters, Samplable)
        nat_cls = exp_parameters.natural_parametrization_cls()  # type: ignore[attr-defined]
        samples = exp_parameters.sample(key, sample_shape)

    dimensions = (exp_parameters.dimensions() if isinstance(exp_parameters, Multidimensional)
                  else 0)
    fixed_parameters = exp_parameters[jnp.newaxis, jnp.newaxis, ...].fixed_parameters()
    support = exp_parameters.domain_support()

    # Verify that the samples have the right shape.
    ideal_shape = (*sample_shape, *distribution_shape, *support.shape(dimensions))
    assert samples.shape == ideal_shape

    # Verify the maximum likelihood estimate.
    sampled_exp_parameters = nat_cls.sufficient_statistics(samples, **fixed_parameters)
    ml_exp_parameters = parameter_mean(sampled_exp_parameters, axis=sample_axes)
    assert_tree_allclose(ml_exp_parameters, exp_parameters, rtol=rtol, atol=atol)


def sample_using_flattened(flattened_parameters: JaxRealArray,
                           parameters_type: type[Samplable],
                           key: KeyArray,
                           fixed_parameters: dict[str, Any]) -> JaxArray:
    parameters = parameters_type.unflattened(flattened_parameters, **fixed_parameters)
    support = parameters.domain_support()
    ordinary_sample = parameters.sample(key)
    return support.flattened(ordinary_sample)


def calculate_jacobian(parameters: Parametrization,
                       keys: KeyArray,
                       ) -> JaxRealArray:
    jacobian_sample = vmap(jacrev(sample_using_flattened, argnums=(0,)), in_axes=(0, None, 0, 0))
    parameters_jacobian, = jacobian_sample(parameters.flattened(), type(parameters), keys,
                                           parameters.fixed_parameters())
    assert parameters_jacobian.shape[0: 1] == parameters.shape
    parameters_jacobian = jnp.sum(parameters_jacobian, axis=0)
    return jnp.sum(parameters_jacobian, axis=0)


def calculate_curvature(parameters: Parametrization,
                        keys: KeyArray,
                        ) -> JaxRealArray:
    # Calculate curvature.
    hessian_sample = vmap(hessian(sample_using_flattened, argnums=(0,)), in_axes=(0, None, 0, 0))
    parameters_hessian_x, = hessian_sample(parameters.flattened(), type(parameters), keys,
                                           parameters.fixed_parameters())
    parameters_hessian, = parameters_hessian_x

    assert parameters_hessian.shape[0: 1] == parameters.shape
    parameters_hessian = jnp.sum(parameters_hessian, axis=0)

    assert parameters_hessian.shape[-2] == parameters_hessian.shape[-1]
    diagonal_hessian = jnp.diagonal(parameters_hessian, axis1=-2, axis2=-1)
    # There's one element for each sample element.  Just sum over the curvatures.
    return jnp.sum(diagonal_hessian, axis=0)


def test_sampling_cotangents(generator: Generator,
                             key: KeyArray,
                             sampling_wc_distribution_info: DistributionInfo[Any, Any, Any],
                             *,
                             distribution_name: None | str,
                             natural: bool) -> None:
    """Test that sampling is consistent with maximum likelihood estimation.

    This tests samples variates from either natural or expectation parametrizations.  Calculates the
    mean of the sufficient statistics, and verifies that it equals the expectation parameters.
    """
    info = sampling_wc_distribution_info
    info.skip_if_deselected(distribution_name)
    distribution_shape = (23,)
    keys = split(key, distribution_shape)

    parameters: Parametrization = (info.nat_parameter_generator(generator, distribution_shape)
                                   if natural
                                   else info.exp_parameter_generator(generator, distribution_shape))
    jacobian = calculate_jacobian(parameters, keys)
    curvature = calculate_curvature(parameters, keys)
    assert jacobian.shape == curvature.shape
    assert jnp.all((jacobian != 0) | (curvature != 0))
