"""Check the conversions among the normal distribution's parametrizations."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import tree

from efax import NormalDP, NormalEP, NormalNP, NormalPP, NormalVP

type _Normal = NormalNP | NormalEP | NormalVP | NormalDP | NormalPP

_CONVERSIONS: dict[type[_Normal], str] = {
    NormalNP: "to_nat",
    NormalEP: "to_exp",
    NormalVP: "to_variance_parametrization",
    NormalDP: "to_deviation_parametrization",
    NormalPP: "to_precision_parametrization",
}


def _distributions() -> tuple[NormalNP, NormalEP, NormalVP, NormalDP, NormalPP]:
    """Return one distribution in every parametrization.

    Build the arrays inside a function so that test collection does not initialize JAX.
    """
    mean = jnp.asarray([-1.5, 0.0, 2.0])
    precision = jnp.asarray([0.25, 1.0, 4.0])
    return (
        NormalNP(mean * precision, -0.5 * precision),
        NormalEP(mean, jnp.square(mean) + 1.0 / precision),
        NormalVP(mean, 1.0 / precision),
        NormalDP(mean, 1.0 / jnp.sqrt(precision)),
        NormalPP(mean, precision),
    )


@pytest.mark.parametrize(
    ("source", "target"), list(product(_CONVERSIONS, repeat=2)), ids=lambda cls: cls.__name__
)
def test_conversion(source: type[_Normal], target: type[_Normal]) -> None:
    distributions = {type(d): d for d in _distributions()}
    distribution = distributions[source]
    converted = distribution if source is target else getattr(distribution, _CONVERSIONS[target])()
    assert isinstance(converted, target)
    for actual, expected in zip(
        tree.leaves(converted), tree.leaves(distributions[target]), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_log_pdf() -> None:
    nat, _, vp, dp, pp = _distributions()
    x = jnp.asarray([-0.7, 0.2, 1.9])
    for distribution in (vp, dp, pp):
        np.testing.assert_allclose(distribution.log_pdf(x), nat.log_pdf(x), rtol=1e-6)


def test_natural_mean() -> None:
    nat, ep, *_ = _distributions()
    np.testing.assert_allclose(nat.mean(), ep.mean)


def test_zero_precision_converts_without_nans() -> None:
    nat = NormalNP(jnp.asarray([0.0, 0.6]), jnp.asarray([0.0, -0.5]))
    pp = nat.to_precision_parametrization()
    np.testing.assert_allclose(pp.mean, [0.0, 0.6])
    np.testing.assert_allclose(pp.precision, [0.0, 1.0])
    np.testing.assert_allclose(pp.to_nat().mean_times_precision, nat.mean_times_precision)
    np.testing.assert_allclose(pp.to_variance_parametrization().variance, [jnp.inf, 1.0])
    gradient = jax.grad(lambda n: jnp.sum(n.mean()))(nat)
    np.testing.assert_allclose(gradient.mean_times_precision, [0.0, 1.0])
    np.testing.assert_allclose(gradient.negative_half_precision, [0.0, 1.2])
    variance_gradient = jax.grad(lambda p: jnp.sum(p.to_variance_parametrization().variance[:1]))
    np.testing.assert_allclose(variance_gradient(pp).precision, [0.0, 0.0])


def test_zero_variance_converts_without_nans() -> None:
    vp = NormalVP(jnp.asarray([0.3, 0.3]), jnp.asarray([0.0, 2.0]))
    np.testing.assert_allclose(vp.to_precision_parametrization().precision, [jnp.inf, 0.5])
    gradient = jax.grad(lambda v: jnp.sum(v.to_precision_parametrization().precision[:1]))(vp)
    np.testing.assert_allclose(gradient.variance, [0.0, 0.0])


def test_variance_cross_entropy() -> None:
    p = NormalVP(jnp.asarray([0.3, -1.0]), jnp.asarray([0.5, 2.0]))
    q = NormalVP(jnp.asarray([-0.2, 0.4]), jnp.asarray([1.5, 0.25]))
    np.testing.assert_allclose(p.cross_entropy(q), p.to_exp().cross_entropy(q.to_nat()))
