from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator
from numpy.testing import assert_allclose
from tjax import NumpyComplexArray, NumpyRealArray, Shape

from efax import ScipyComplexMultivariateNormal, ScipyComplexNormal


# Tools --------------------------------------------------------------------------------------------
def random_complex_array(generator: Generator, shape: Shape = ()) -> NumpyComplexArray:
    return np.asarray(sum(x * generator.normal(size=shape) for x in (0.5, 0.5j)))


def build_uvcn(generator: Generator, shape: Shape) -> ScipyComplexNormal:
    mean = random_complex_array(generator, shape)
    variance = generator.exponential(size=shape)
    pseudo_variance = (variance
                       * generator.beta(2, 2, size=shape)
                       * np.exp(1j * generator.uniform(0, 2 * np.pi, size=shape)))
    return ScipyComplexNormal(mean, variance, pseudo_variance)


def build_mvcn(generator: Generator,
               shape: Shape,
               dimensions: int,
               polarization: float = 0.98
               ) -> ScipyComplexMultivariateNormal:
    directions = 3
    weights = np.asarray(range(directions)) + 1.5
    mean = random_complex_array(generator, (*shape, dimensions))
    z = random_complex_array(generator, (*shape, dimensions, directions))
    regularizer = np.tile(np.eye(dimensions), (*shape, 1, 1))
    variance = (
        np.average(
            np.conj(z)[..., np.newaxis, :, :] * z[..., np.newaxis, :],  # type: ignore[arg-type]
            weights=weights,
            axis=-1)
        + regularizer)
    pseudo_variance = np.average(z[..., np.newaxis, :, :] * z[..., np.newaxis, :],
                                 weights=weights,
                                 axis=-1) * polarization
    return ScipyComplexMultivariateNormal(mean, variance, pseudo_variance)


# Tests --------------------------------------------------------------------------------------------
@pytest.mark.nondistribution
def test_univariate_rvs(generator: Generator) -> None:
    shape = (3, 2)
    rvs_shape = (50, 100)
    rvs_axes = tuple(range(-len(rvs_shape), 0))
    dist = build_uvcn(generator, shape)
    rvs = dist.rvs(random_state=generator, size=rvs_shape)
    assert rvs.shape == shape + rvs_shape

    estimated_mean = np.average(rvs, axis=rvs_axes)
    centered_rvs = rvs - dist.mean[(...,) + (np.newaxis,) * len(rvs_shape)]
    estimated_variance = np.real(np.average(
        np.conj(centered_rvs) * centered_rvs, axis=rvs_axes))
    estimated_pseudo_variance = np.average(np.square(centered_rvs), axis=rvs_axes)
    assert_allclose(estimated_mean, dist.mean, rtol=0.0, atol=2e-2)
    assert_allclose(estimated_variance, dist.variance, rtol=3e-2, atol=2e-2)
    assert_allclose(estimated_pseudo_variance, dist.pseudo_variance, rtol=3e-2,
                    atol=2e-2)


@pytest.mark.nondistribution
def test_multivariate_rvs(generator: Generator) -> None:
    shape = (3, 5)
    rvs_shape = (256, 512)
    dimensions = 2
    rvs_axes = tuple(range(-len(rvs_shape) - 1, -1))
    rvs_axes2 = tuple(range(-len(rvs_shape) - 2, -2))
    dist = build_mvcn(generator, shape, dimensions)
    rvs = dist.rvs(random_state=generator, size=rvs_shape)
    assert rvs.shape == shape + rvs_shape + (dimensions,)

    estimated_mean = np.average(rvs, axis=rvs_axes)
    centered_rvs = rvs - dist.mean[(...,) + (np.newaxis,) * len(rvs_shape) + (slice(None),)]
    estimated_variance = np.real(np.average(
        centered_rvs[..., np.newaxis] * np.conj(centered_rvs)[..., np.newaxis, :],
        axis=rvs_axes2))
    estimated_pseudo_variance = np.average(
        centered_rvs[..., np.newaxis] * centered_rvs[..., np.newaxis, :], axis=rvs_axes2)
    assert_allclose(estimated_mean, dist.mean, rtol=1e-2, atol=1e-2)
    assert_allclose(estimated_variance, dist.variance, rtol=2e-1, atol=5e-1)
    assert_allclose(estimated_pseudo_variance, dist.pseudo_variance, rtol=2e-1,
                    atol=5e-1)


@pytest.mark.nondistribution
def test_univariate_multivariate_consistency(generator: Generator) -> None:
    mv = build_mvcn(generator, (), 1, polarization=0.5)
    component = mv.access_object(())
    mean: NumpyComplexArray = np.asarray(component.mean[0])
    variance: NumpyRealArray = np.real(np.asarray(component.variance[0, 0]))
    pseudo_variance: NumpyComplexArray = np.asarray(component.pseudo_variance[0, 0])
    uv = ScipyComplexNormal(mean, variance, pseudo_variance)
    x = random_complex_array(generator)
    assert_allclose(mv.pdf(x[np.newaxis]), uv.pdf(x))
