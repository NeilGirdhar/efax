"""Tests for NaturalParametrization.characteristic_function (CF of sufficient statistics)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate as si
import scipy.special as sc
from numpy.random import Generator
from numpy.testing import assert_allclose

from efax import (
    ExponentialNP,
    GammaNP,
    MultivariateDiagonalNormalNP,
    NormalNP,
    PoissonNP,
    VonMisesFisherNP,
    parameter_map,
)

from .distribution_info import DistributionInfo


def test_cf_jacfwd_matches_finite_differences() -> None:
    """Check jax.jacfwd CF derivative against central difference.

    For Exponential (1-D natural param), compare jax.jacfwd of CF to a
    central-difference reference. The CF is rate / (rate - i·f); its derivative
    in f at f=0 is i / rate.
    """
    rate = 1.7
    p = ExponentialNP(jnp.float64(-rate))
    t_zero = ExponentialNP(jnp.float64(0.0))

    def cf(t: ExponentialNP) -> jnp.ndarray:
        return p.characteristic_function(t)

    jac = jax.jacfwd(cf)(t_zero)
    # Pull the (sole) leaf out as a complex scalar.
    (leaf,) = jax.tree_util.tree_leaves(jac)
    eps = 1e-3
    plus = cf(ExponentialNP(jnp.float64(eps)))
    minus = cf(ExponentialNP(jnp.float64(-eps)))
    fd = (plus - minus) / (2 * eps)
    assert_allclose(complex(leaf), complex(fd), rtol=1e-4, atol=1e-6)
    # Sanity check against the analytic value i / rate.
    assert_allclose(complex(leaf), 1j / rate, rtol=1e-4, atol=1e-6)


def _normal_cf_x2(mu: float, sigma2: float, g: float) -> complex:
    """E[exp(i*g*x^2)] for x ~ N(mu, sigma^2) via numerical integration."""

    def re(x: float) -> float:
        return float(np.cos(g * x**2) * np.exp(-((x - mu) ** 2) / (2 * sigma2)))

    def im(x: float) -> float:
        return float(np.sin(g * x**2) * np.exp(-((x - mu) ** 2) / (2 * sigma2)))

    scale = np.sqrt(2 * np.pi * sigma2)
    return complex(si.quad(re, -np.inf, np.inf)[0], si.quad(im, -np.inf, np.inf)[0]) / scale


def test_cf_at_zero(distribution_info: DistributionInfo, generator: Generator) -> None:
    """CF(t=0) = 1 for all distributions."""
    p = distribution_info.nat_parameter_generator(generator, shape=())
    zero = parameter_map(jnp.zeros_like, p)
    cf = p.characteristic_function(zero)
    assert_allclose(complex(cf), 1.0 + 0j, atol=1e-6)


def test_normal_cf_of_x() -> None:
    # CF of T(x)=(x, x^2) at t=(f, 0) reduces to E[exp(i*f*x)] = exp(i*mu*f - sigma^2*f^2/2).
    rng = np.random.default_rng(0)
    for _ in range(10):
        mtp = rng.normal(0, 1)
        nhp = -rng.uniform(0.1, 2)
        f = rng.normal(0, 1)
        p = NormalNP(jnp.float64(mtp), jnp.float64(nhp))
        cf = p.characteristic_function(NormalNP(jnp.float64(f), jnp.float64(0.0)))
        mu = -mtp / (2 * nhp)
        sigma2 = -1 / (2 * nhp)
        assert_allclose(complex(cf), np.exp(1j * mu * f - sigma2 * f**2 / 2), rtol=1e-5)


def test_normal_cf_of_x_squared() -> None:
    # CF of T(x)=(x, x^2) at t=(0, g): E[exp(i*g*x^2)], verified by numerical integration.
    rng = np.random.default_rng(1)
    for _ in range(5):
        mtp = rng.normal(0, 0.5)
        nhp = -rng.uniform(0.2, 1.5)
        g = rng.uniform(-0.3, 0.3)
        p = NormalNP(jnp.float64(mtp), jnp.float64(nhp))
        cf = p.characteristic_function(NormalNP(jnp.float64(0.0), jnp.float64(g)))
        mu = -mtp / (2 * nhp)
        sigma2 = -1 / (2 * nhp)
        assert_allclose(complex(cf), _normal_cf_x2(mu, sigma2, g), rtol=1e-5)


def test_poisson_cf() -> None:
    # Poisson T(x) = x, so CF of T(x) = CF of x = exp(lambda*(exp(i*f) - 1)).
    rng = np.random.default_rng(2)
    for _ in range(10):
        lam = rng.uniform(0.5, 5)
        f = rng.normal(0, 1)
        p = PoissonNP(jnp.float64(np.log(lam)))
        cf = p.characteristic_function(PoissonNP(jnp.float64(f)))
        assert_allclose(complex(cf), np.exp(lam * (np.exp(1j * f) - 1)), rtol=1e-5)


def test_exponential_cf() -> None:
    # Exponential T(x) = x, so CF of T(x) = rate / (rate - i*f).
    rng = np.random.default_rng(3)
    for _ in range(10):
        rate = rng.uniform(0.5, 3)
        f = rng.normal(0, 1)
        p = ExponentialNP(jnp.float64(-rate))
        cf = p.characteristic_function(ExponentialNP(jnp.float64(f)))
        assert_allclose(complex(cf), rate / (rate - 1j * f), rtol=1e-5)


def test_gamma_cf_of_x_and_log_x() -> None:
    # Gamma T(x) = (x, log x).
    rng = np.random.default_rng(4)
    for _ in range(10):
        rate = rng.uniform(0.5, 3)
        alpha = rng.uniform(0.5, 3)
        f = rng.normal(0, 1)
        g = rng.uniform(-0.3, 0.3)
        p = GammaNP(jnp.float64(-rate), jnp.float64(alpha - 1))
        cf = p.characteristic_function(GammaNP(jnp.float64(f), jnp.float64(g)))
        expected = np.exp(
            sc.loggamma(alpha + 1j * g)
            - sc.loggamma(alpha)
            + alpha * np.log(rate)
            - (alpha + 1j * g) * np.log(rate - 1j * f)
        )
        assert_allclose(complex(cf), expected, rtol=1e-5)


def test_von_mises_fisher_cf_derivative_recovers_mean() -> None:
    p = VonMisesFisherNP(jnp.asarray([1.2, -0.7], dtype=jnp.float64))
    t_zero = VonMisesFisherNP(jnp.zeros(2, dtype=jnp.float64))

    jac = jax.jacfwd(p.characteristic_function)(t_zero)

    assert_allclose(
        np.asarray(jac.mean_times_concentration),
        1j * np.asarray(p.to_exp().mean),
        rtol=1e-5,
    )


def test_multivariate_diagonal_normal_cf() -> None:
    # T(x) = (x, x^2) elementwise.
    # t=(f, 0): E[exp(i*f^T*x)] = exp(i*f^T*mu - 0.5*sum(f^2*sigma^2))
    # t=(0, g): E[exp(i*sum(g*x^2))] = product of per-component integrals
    rng = np.random.default_rng(5)
    d = 4
    for _ in range(5):
        precision = rng.uniform(0.5, 3, d)
        mu = rng.normal(0, 1, d)
        f = rng.normal(0, 1, d)
        g = rng.uniform(-0.3, 0.3, d)
        sigma2 = 1 / precision
        p = MultivariateDiagonalNormalNP(jnp.array(precision * mu), jnp.array(-0.5 * precision))

        cf_x = p.characteristic_function(MultivariateDiagonalNormalNP(jnp.array(f), jnp.zeros(d)))
        assert_allclose(
            complex(cf_x),
            np.exp(1j * f @ mu - 0.5 * (f**2 @ sigma2)),
            rtol=1e-5,
        )

        cf_x2 = p.characteristic_function(MultivariateDiagonalNormalNP(jnp.zeros(d), jnp.array(g)))
        expected_x2 = np.prod([_normal_cf_x2(mu[k], sigma2[k], g[k]) for k in range(d)])
        assert_allclose(complex(cf_x2), expected_x2, rtol=1e-4)
