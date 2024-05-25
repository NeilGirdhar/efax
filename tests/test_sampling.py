"""These tests apply to only samplable distributions."""
from __future__ import annotations

from typing import Any, TypeAlias

import jax.numpy as jnp
from numpy.random import Generator
from tjax import JaxComplexArray, KeyArray, Shape, assert_tree_allclose

from efax import (ExpectationParametrization, JointDistribution, NaturalParametrization, Samplable,
                  SimpleDistribution, Structure, fixed_parameter_packet, flatten_mapping,
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
                           if issubclass(info.type_, SimpleDistribution)}
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
        if not issubclass(info.type_, SimpleDistribution):
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
