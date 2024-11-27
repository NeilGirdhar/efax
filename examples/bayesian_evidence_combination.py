"""Bayesian evidence combination.

This example is based on section 1.2.1 from expfam.pdf, entitled Bayesian evidence combination.

Suppose you have a prior, and a set of likelihoods, and you want to combine all of the evidence
into one distribution.
"""
from operator import add

import jax.numpy as jnp
from tjax import print_generic

from efax import MultivariateDiagonalNormalVP, parameter_map

prior = MultivariateDiagonalNormalVP(mean=jnp.zeros(2),
                                     variance=10 * jnp.ones(2))
likelihood = MultivariateDiagonalNormalVP(mean=jnp.asarray([1.1, -2.2]),
                                          variance=jnp.asarray([3.0, 1.0]))

# Convert to the natural parametrization.
prior_np = prior.to_nat()
likelihood_np = likelihood.to_nat()

# Sum.  We use parameter_map to ensure that we don't accidentally add "fixed" parameters, e.g., the
# failure count of a negative binomial distribution.
posterior_np = parameter_map(add, prior_np, likelihood_np)

# Convert to the source parametrization.
posterior = posterior_np.to_variance_parametrization()
print_generic(posterior)
# MultivariateDiagonalNormalVP[dataclass]
# ├── mean=Jax Array (2,) float32
# │   └──  0.0355 │ -0.2000
# └── variance=Jax Array (2,) float32
#     └──  0.0968 │ 0.0909
