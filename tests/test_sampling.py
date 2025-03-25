"""These tests apply to only samplable distributions."""
from __future__ import annotations

from typing import Any, TypeAlias

import jax.numpy as jnp
from numpy.random import Generator
from tjax import JaxComplexArray, KeyArray, Shape, assert_tree_allclose

from efax import (Distribution, ExpectationParametrization, JointDistribution,
                  MaximumLikelihoodEstimator, NaturalParametrization, Samplable, SimpleDistribution,
                  Structure, flat_dict_of_observations, flatten_mapping, parameter_mean)

from .create_info import (ComplexCircularlySymmetricNormalInfo,
                          ComplexMultivariateUnitVarianceNormalInfo, ComplexNormalInfo,
                          IsotropicNormalInfo, JointInfo, MultivariateDiagonalNormalInfo,
                          MultivariateFixedVarianceNormalInfo, MultivariateNormalInfo,
                          MultivariateUnitVarianceNormalInfo, PoissonInfo)
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
    sampling_object: Distribution
    if natural:
        sampling_object = nat_parameters = sampling_distribution_info.nat_parameter_generator(
                generator, distribution_shape)
        exp_parameters = nat_parameters.to_exp()
    else:
        sampling_object = exp_parameters = sampling_distribution_info.exp_parameter_generator(
                generator, distribution_shape)

    samples: dict[str, Any] | JaxComplexArray
    if isinstance(sampling_object, Samplable):
        unclamped_samples = sampling_object.sample(key, sample_shape)
        samples = sampling_object.domain_support().clamp(unclamped_samples)
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
                           for info in structure.infos
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
        samples: dict[str, Any] | JaxComplexArray
        ) -> None:
    atol = (1e-2
            if isinstance(sampling_distribution_info,
                          ComplexCircularlySymmetricNormalInfo
                          | ComplexMultivariateUnitVarianceNormalInfo
                          | ComplexNormalInfo
                          | IsotropicNormalInfo
                          | JointInfo
                          | MultivariateDiagonalNormalInfo
                          | MultivariateFixedVarianceNormalInfo
                          | MultivariateNormalInfo
                          | MultivariateUnitVarianceNormalInfo)
            else 1e-3
            if isinstance(sampling_distribution_info, JointInfo | PoissonInfo)
            else 1e-6)
    rtol = 4e-2
    sample_axes = tuple(range(len(sample_shape)))
    newaxes = (jnp.newaxis,) * len(sample_shape)
    estimator = MaximumLikelihoodEstimator.create_estimator(exp_parameters[*newaxes, ...])
    sampled_exp_parameters = estimator.sufficient_statistics(samples)
    maximum_likelihood_parameters = parameter_mean(sampled_exp_parameters, axis=sample_axes)
    assert_tree_allclose(maximum_likelihood_parameters, exp_parameters, rtol=rtol,
                         atol=atol)


def test_sampling_and_estimation(generator: Generator,
                                 key: KeyArray,
                                 sampling_distribution_info: DistributionInfo[Any, Any, Any],
                                 *,
                                 distribution_name: str | None,
                                 natural: bool) -> None:
    """Test that sampling is consistent with maximum likelihood estimation.

    This tests samples variates from either natural or expectation parametrizations.  Calculates the
    mean of the sufficient statistics, and verifies that it equals the expectation parameters.
    """
    sampling_distribution_info.skip_if_deselected(distribution_name)
    distribution_shape = (4,)  # The number of distributions that are being estimated.
    sample_shape = (1024, 64)  # The number of samples that are taken to do the estimation.
    exp_parameters, samples = produce_samples(generator, key, sampling_distribution_info,
                                              distribution_shape, sample_shape, natural=natural)
    flat_map_of_samples = flat_dict_of_observations(samples)
    structure = Structure.create(exp_parameters)
    flat_map_of_samples = flatten_mapping(samples) if isinstance(samples, dict) else {(): samples}
    verify_sample_shape(distribution_shape, sample_shape, structure, flat_map_of_samples)
    verify_maximum_likelihood_estimate(sampling_distribution_info, sample_shape, structure,
                                       exp_parameters, samples)
