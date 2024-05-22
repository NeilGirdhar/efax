"""These tests apply to only samplable distributions."""
from __future__ import annotations

from typing import Any, TypeAlias

import jax.numpy as jnp
from jax import jacrev, vmap
from jax.random import split
from numpy.random import Generator
from tjax import (JaxArray, JaxComplexArray, JaxRealArray, KeyArray, Shape, assert_tree_allclose,
                  hessian)

from efax import (ExpectationParametrization, Flattener, JointDistribution, NaturalParametrization,
                  Parametrization, Samplable, Structure, fixed_parameter_packet, flatten_mapping,
                  parameter_mean)

from .create_info import (ComplexCircularlySymmetricNormalInfo, ComplexMultivariateUnitNormalInfo,
                          ComplexNormalInfo, GammaInfo, IsotropicNormalInfo,
                          MultivariateDiagonalNormalInfo, MultivariateFixedVarianceNormalInfo,
                          MultivariateNormalInfo, MultivariateUnitNormalInfo, PoissonInfo)
from .distribution_info import DistributionInfo

Path: TypeAlias = tuple[str, ...]


def produce_samples(generator: Generator,
                    key: KeyArray,
                    sampling_distribution_info: DistributionInfo[NaturalParametrization[Any, Any],
                                                                 ExpectationParametrization[Any],
                                                                 Any],
                    distribution_shape: Shape,
                    sample_shape: Shape,
                    *,
                    natural: bool) -> tuple[ExpectationParametrization[Any],
                                            dict[str, Any] | JaxComplexArray]:
    if natural:
        sampling_object = nat_parameters = sampling_distribution_info.nat_parameter_generator(
                generator, distribution_shape)
        exp_parameters = nat_parameters.to_exp()
    else:
        sampling_object = exp_parameters = sampling_distribution_info.exp_parameter_generator(
                generator, distribution_shape)

    if isinstance(sampling_object, Samplable):
        samples = sampling_object.sample(key, sample_shape)
    else:
        assert isinstance(sampling_object, JointDistribution)
        samples = sampling_object.general_sample(key, sample_shape)
    return exp_parameters, samples


def verify_sample_shape(distribution_shape: Shape,
                        sample_shape: Shape,
                        structure: Structure[ExpectationParametrization[Any]],
                        flat_map_of_samples: dict[Path, Any]
                        ) -> None:
    ideal_samples_shape = {info.path: (*sample_shape, *distribution_shape,
                                       *info.type_.domain_support().shape(info.dimensions))
                           for info in structure.distributions
                           if issubclass(info.type_, Parametrization)}
    samples_shape = {path: s.shape for path, s in flat_map_of_samples.items()}
    assert samples_shape == ideal_samples_shape


def verify_maximum_likelihood_estimate(
        sampling_distribution_info: DistributionInfo[NaturalParametrization[Any, Any],
                                                     ExpectationParametrization[Any],
                                                     Any],
        sample_shape: Shape,
        structure: Structure[ExpectationParametrization[Any]],
        exp_parameters: ExpectationParametrization[Any],
        flat_map_of_samples: dict[Path, Any]
        ) -> None:
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
    sample_axes = tuple(range(len(sample_shape)))
    newaxes = (jnp.newaxis,) * len(sample_shape)
    fixed_parameters = fixed_parameter_packet(exp_parameters[*newaxes, ...])
    flat_map_of_parameters: dict[Path, ExpectationParametrization[Any]] = (
            flatten_mapping(exp_parameters.as_dict())
            if isinstance(exp_parameters, JointDistribution)
            else {(): exp_parameters})
    for info in structure.distributions:
        if not issubclass(info.type_, Parametrization):
            continue
        assert issubclass(info.type_, ExpectationParametrization)
        nat_cls = info.type_.natural_parametrization_cls()
        samples = flat_map_of_samples[info.path]
        sampled_exp_parameters = nat_cls.sufficient_statistics(samples, **fixed_parameters)
        maximum_likelihood_parameters = parameter_mean(sampled_exp_parameters, axis=sample_axes)
        observed_parameters = flat_map_of_parameters[info.path]
        assert_tree_allclose(maximum_likelihood_parameters, observed_parameters, rtol=rtol,
                             atol=atol)


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
    exp_parameters, samples = produce_samples(generator, key, sampling_distribution_info,
                                              distribution_shape, sample_shape, natural=natural)
    flat_map_of_samples = flatten_mapping(samples) if isinstance(samples, dict) else {(): samples}
    structure = Structure.create(exp_parameters)
    verify_sample_shape(distribution_shape, sample_shape, structure, flat_map_of_samples)
    verify_maximum_likelihood_estimate(sampling_distribution_info, sample_shape, structure,
                                       exp_parameters, flat_map_of_samples)


# TODO: Move to new file.
def sample_using_flattened(flattened_parameters: JaxRealArray,
                           flattener: Flattener[Any],
                           key: KeyArray,
                           ) -> JaxArray:
    p = flattener.unflatten(flattened_parameters)

    def flattened_sample(q: Parametrization) -> JaxRealArray:
        assert isinstance(q, Samplable)
        ordinary_sample = q.sample(key)
        support = q.domain_support()
        return support.flattened(ordinary_sample, map_to_plane=False)

    if isinstance(p, Parametrization):
        return flattened_sample(p)
    sample_dict = p.general_method(flattened_sample)
    sample_flat = flatten_mapping(sample_dict)
    return jnp.concatenate(list(sample_flat.values()), axis=-1)


def calculate_jacobian(p: Parametrization,
                       keys: KeyArray,
                       ) -> JaxRealArray:
    flattener, flattened = Flattener.flatten(p)
    jacobian_sample = vmap(jacrev(sample_using_flattened, argnums=(0,)), in_axes=(0, 0, 0))
    parameters_jacobian, = jacobian_sample(flattened, flattener, keys)
    assert parameters_jacobian.shape[0: 1] == p.shape
    parameters_jacobian = jnp.sum(parameters_jacobian, axis=0)
    return jnp.sum(parameters_jacobian, axis=0)


def calculate_curvature(p: Parametrization,
                        keys: KeyArray,
                        ) -> JaxRealArray:
    # Calculate curvature.
    flattener, flattened = Flattener.flatten(p)
    hessian_sample = vmap(hessian(sample_using_flattened, argnums=(0,), reverse_only=True),
                          in_axes=(0, 0, 0))
    parameters_hessian_x, = hessian_sample(flattened, flattener, keys)
    parameters_hessian, = parameters_hessian_x

    assert parameters_hessian.shape[0: 1] == p.shape
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
    """Test that the curvature is nonzero."""
    info = sampling_wc_distribution_info
    info.skip_if_deselected(distribution_name)
    distribution_shape = (23,)
    keys = split(key, distribution_shape)

    p: Parametrization = (info.nat_parameter_generator(generator, distribution_shape)
                          if natural
                          else info.exp_parameter_generator(generator, distribution_shape))
    jacobian = calculate_jacobian(p, keys)
    curvature = calculate_curvature(p, keys)
    assert jacobian.shape == curvature.shape
    assert jnp.all((jacobian != 0) | (curvature != 0))
