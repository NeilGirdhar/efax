"""These tests apply to only samplable distributions."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
from jax import jacobian, jit, vmap
from numpy.random import Generator
from tjax import JaxArray, JaxRealArray, KeyArray, RngStream, hessian

from efax import Flattener, Samplable, SimpleDistribution, flatten_mapping

from .distribution_info import DistributionInfo
from .shapes import DIST_SHAPE_MEDIUM


def _sample_using_flattened(
    flattened_parameters: JaxRealArray,
    flattener: Flattener,
    key: KeyArray,
) -> JaxArray:
    p = flattener.unflatten(flattened_parameters)
    stream = RngStream(key)

    def flattened_sample(q: SimpleDistribution) -> JaxRealArray:
        assert isinstance(q, Samplable)
        ordinary_sample = q.sample(stream.key())
        support = q.domain_support()
        return support.flattened(ordinary_sample, map_to_plane=False)

    if isinstance(p, SimpleDistribution):
        return flattened_sample(p)
    sample_dict = p.general_method(flattened_sample)
    sample_flat = flatten_mapping(sample_dict)
    return jnp.concat(tuple(sample_flat.values()), axis=-1)


def _calculate_jacobian(
    p: SimpleDistribution,
    keys: KeyArray,
) -> JaxRealArray:
    flattener, flattened = Flattener.flatten(p)
    jacobian_sample = jacobian(_sample_using_flattened, argnums=(0,))
    for _ in p.shape:
        jacobian_sample = vmap(jacobian_sample, in_axes=(0, 0, 0))
    (parameters_jacobian,) = jacobian_sample(flattened, flattener, keys)
    assert parameters_jacobian.shape[: len(p.shape)] == p.shape
    return jnp.sum(parameters_jacobian, axis=tuple(range(len(p.shape) + 1)))


def _calculate_curvature(
    p: SimpleDistribution,
    keys: KeyArray,
) -> JaxRealArray:
    # Calculate curvature.
    flattener, flattened = Flattener.flatten(p)
    hessian_sample = hessian(_sample_using_flattened, argnums=(0,))
    for _ in p.shape:
        hessian_sample = vmap(hessian_sample, in_axes=(0, 0, 0))
    (parameters_hessian_x,) = hessian_sample(flattened, flattener, keys)
    (parameters_hessian,) = parameters_hessian_x

    assert parameters_hessian.shape[: len(p.shape)] == p.shape
    parameters_hessian = jnp.sum(parameters_hessian, axis=tuple(range(len(p.shape))))

    assert parameters_hessian.shape[-2] == parameters_hessian.shape[-1]
    diagonal_hessian = jnp.linalg.diagonal(parameters_hessian)
    # There's one element for each sample element.  Just sum over the curvatures.
    return jnp.sum(diagonal_hessian, axis=0)


@jit
def _calculate_jacobian_and_curvature(
    p: SimpleDistribution,
    jacobian_keys: KeyArray,
    curvature_keys: KeyArray,
) -> tuple[JaxRealArray, JaxRealArray]:
    return _calculate_jacobian(p, jacobian_keys), _calculate_curvature(p, curvature_keys)


def test_sampling_cotangents(
    generator: Generator,
    key: KeyArray,
    sampling_wc_distribution_info: DistributionInfo,
    *,
    distribution_name: str | None,
    natural: bool,
) -> None:
    """Test that the curvature is nonzero."""
    info = sampling_wc_distribution_info
    info.skip_if_deselected(distribution_name)
    distribution_shape = DIST_SHAPE_MEDIUM
    jacobian_keys, curvature_keys = jr.split(key, (2, *distribution_shape))
    p: SimpleDistribution = (
        info.nat_parameter_generator(generator, distribution_shape)
        if natural
        else info.exp_parameter_generator(generator, distribution_shape)
    )
    jacobian, curvature = _calculate_jacobian_and_curvature(p, jacobian_keys, curvature_keys)
    assert jacobian.shape == curvature.shape
    assert jnp.all((jacobian != 0) | (curvature != 0))
