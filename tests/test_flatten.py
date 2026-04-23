from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.random import Generator
from tjax import Shape, assert_tree_allclose
from tjax.dataclasses import dataclass

from efax import (
    Flattener,
    MultivariateUnitVarianceNormalNP,
    ScalarSupport,
    SimpleDistribution,
    distribution_parameter,
)

from .distribution_info import DistributionInfo


@dataclass
class FixedOnlyDistribution(SimpleDistribution):
    value: jnp.ndarray = distribution_parameter(ScalarSupport(), fixed=True)

    @property
    def shape(self) -> Shape:
        return self.value.shape

    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport()


@pytest.mark.parametrize("natural", [False, True])
def test_flatten(
    generator: Generator,
    distribution_info: DistributionInfo,
    *,
    natural: bool,
) -> None:
    """Test that unflattening restores the source of flattening."""
    shape = (3, 4)
    p = (
        distribution_info.exp_parameter_generator(generator, shape=shape)
        if natural
        else distribution_info.nat_parameter_generator(generator, shape=shape)
    )
    flattener, flattened = Flattener.flatten(p)
    assert issubclass(flattened.dtype.type, np.floating)
    assert_tree_allclose(flattener.unflatten(flattened), p)


@pytest.mark.nondistribution
def test_unflatten_raveled() -> None:
    """Test that raveled=True accepts a 1D raveled array and matches the normal path."""
    m = MultivariateUnitVarianceNormalNP(jnp.zeros((3, 10)))
    flattener, flattened = Flattener.flatten(m)
    assert flattened.shape == (3, 10)
    raveled = jnp.ravel(flattened)
    assert raveled.shape == (30,)
    assert_tree_allclose(flattener.unflatten(raveled, raveled=True), m)


@pytest.mark.nondistribution
def test_flatten_fixed_only_distribution() -> None:
    p = FixedOnlyDistribution(jnp.arange(6.0).reshape(2, 3))

    flattener, flattened = Flattener.flatten(p)

    assert flattened.shape == (2, 3, 0)
    assert flattener.final_dimension_size() == 0
    assert_tree_allclose(flattener.unflatten(flattened), p)
    assert_tree_allclose(flattener.unflatten(jnp.zeros(0), raveled=True), p)


@pytest.mark.nondistribution
def test_raises() -> None:
    """Test that unflattening raises for bad shapes."""
    m = MultivariateUnitVarianceNormalNP(jnp.zeros(10))
    flattener, flattened = Flattener.flatten(m)
    assert flattened.shape == (10,)
    with pytest.raises(ValueError, match="Incompatible array"):
        flattener.unflatten(jnp.zeros(5))
    with pytest.raises(ValueError, match="Incompatible array"):
        flattener.unflatten(jnp.zeros(12))
