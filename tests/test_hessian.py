"""These tests apply to only samplable distributions."""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import jacrev, vmap
from jax.random import split
from numpy.random import Generator
from tjax import JaxArray, JaxRealArray, KeyArray, hessian

from efax import Flattener, Samplable, SimpleDistribution, flatten_mapping

from .distribution_info import DistributionInfo


def sample_using_flattened(flattened_parameters: JaxRealArray,
                           flattener: Flattener[Any],
                           key: KeyArray,
                           ) -> JaxArray:
    p = flattener.unflatten(flattened_parameters)

    def flattened_sample(q: SimpleDistribution) -> JaxRealArray:
        assert isinstance(q, Samplable)
        ordinary_sample = q.sample(key)
        support = q.domain_support()
        return support.flattened(ordinary_sample, map_to_plane=False)

    if isinstance(p, SimpleDistribution):
        return flattened_sample(p)
    sample_dict = p.general_method(flattened_sample)
    sample_flat = flatten_mapping(sample_dict)
    return jnp.concatenate(list(sample_flat.values()), axis=-1)


def calculate_jacobian(p: SimpleDistribution,
                       keys: KeyArray,
                       ) -> JaxRealArray:
    flattener, flattened = Flattener.flatten(p)
    jacobian_sample = vmap(jacrev(sample_using_flattened, argnums=(0,)), in_axes=(0, 0, 0))
    parameters_jacobian, = jacobian_sample(flattened, flattener, keys)
    assert parameters_jacobian.shape[0: 1] == p.shape
    parameters_jacobian = jnp.sum(parameters_jacobian, axis=0)
    return jnp.sum(parameters_jacobian, axis=0)


def calculate_curvature(p: SimpleDistribution,
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

    p: SimpleDistribution = (info.nat_parameter_generator(generator, distribution_shape)
                          if natural
                          else info.exp_parameter_generator(generator, distribution_shape))
    jacobian = calculate_jacobian(p, keys)
    curvature = calculate_curvature(p, keys)
    assert jacobian.shape == curvature.shape
    assert jnp.all((jacobian != 0) | (curvature != 0))
