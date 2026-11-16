[![PyPI - Version](https://img.shields.io/pypi/v/efax)](https://pypi.org/project/efax/)
[![EffVer Versioning](https://img.shields.io/badge/version_scheme-EffVer-0097a7)](https://jacobtomlinson.dev/effver)
[![SPEC-0](https://img.shields.io/badge/SPEC-0-blue)](https://scientific-python.org/specs/spec-0000/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/efax)

# EFAX: Exponential Families in JAX

EFAX gives every exponential family distribution its two natural coordinate systems, the natural
and expectation parametrizations, behind one interface that works with JAX's `jit`, `grad`, and
`vmap`.

The two parametrizations matter because the most common operations in probabilistic machine
learning are simple in one of them. Independent evidence adds in natural parameters, maximum
likelihood estimates are averages in expectation parameters, and cross entropy is cheapest
between an expectation parametrization and a natural one. EFAX converts between them, so each
operation stays a line of code:

```python
from operator import add

import jax
import jax.numpy as jnp
import jax.random as jr

from efax import (
    BernoulliEP,
    BernoulliNP,
    DirichletEP,
    DirichletNP,
    Estimator,
    NormalVP,
    parameter_map,
)

# Bayesian updating: add the prior's and the likelihood's natural parameters.
prior = NormalVP(mean=jnp.asarray(0.0), variance=jnp.asarray(10.0))
likelihood = NormalVP(mean=jnp.asarray(1.1), variance=jnp.asarray(3.0))
posterior = parameter_map(add, prior.to_nat(), likelihood.to_nat())
posterior.to_variance_parametrization()  # mean 0.846, variance 2.308

# Maximum likelihood estimation: average the sufficient statistics.
samples = DirichletNP(alpha_minus_one=jnp.asarray([1.0, 2.0, 3.0])).sample(jr.key(0), (10_000,))
Estimator.from_type(DirichletEP).mle(samples, axis=0).to_nat()  # alpha - 1 ≈ 1.02, 2.05, 2.99

# Cross entropy, differentiated with respect to the prediction's log-odds.
target = BernoulliEP(jnp.asarray([0.3, 0.4, 0.7]))
prediction = BernoulliNP(jnp.zeros(3))
jax.grad(lambda q: jnp.sum(target.cross_entropy(q)))(prediction)  # BernoulliNP: 0.2, 0.1, -0.2
```

## Installation

```shell
pip install efax
```

## Features

- More than 30 distributions, from the normal, gamma, and Dirichlet distributions to the von
  Mises-Fisher and Wishart distributions.
- Vectorized objects: one object of shape `(1000,)` holds a thousand distributions.
- Pytrees throughout, so distributions pass through `jit`, `grad`, `vmap`, and `lax` loops.
- Entropy, cross entropy, KL divergence, densities, and sampling.
- Maximum likelihood estimation and conjugate priors.
- Fisher information and Jeffreys prior densities.
- Joint distributions that combine independent distributions into one object.
- Flattening of distributions into unconstrained arrays for neural networks and optimizers.

## Documentation

- [Concepts](https://github.com/NeilGirdhar/efax/blob/main/docs/concepts.md): how EFAX
  represents distributions, its parametrizations, and its methods.
- [Distributions](https://github.com/NeilGirdhar/efax/blob/main/docs/distributions.md): the
  supported families.
- [Structure utilities](https://github.com/NeilGirdhar/efax/blob/main/docs/structure.md):
  assembling, estimating, flattening, and operating on parameters.
- [Examples](https://github.com/NeilGirdhar/efax/blob/main/docs/examples.md): cross entropy,
  evidence combination, optimization, and maximum likelihood estimation.
- [Overview of exponential families](https://github.com/NeilGirdhar/efax/blob/main/exponential_families.pdf):
  the mathematics behind the library.

## Contributing and citing

Contributions are welcome, including new distributions, which usually take about a hundred lines.
See the [contribution guidelines](https://github.com/NeilGirdhar/efax/blob/main/CONTRIBUTING.md).

To cite EFAX, use the "Cite this repository" button on
[GitHub](https://github.com/NeilGirdhar/efax), which reads
[CITATION.cff](https://github.com/NeilGirdhar/efax/blob/main/CITATION.cff).
