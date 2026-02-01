"""Maximum likelihood estimation.

This example is based on section 1.3.2 from exponential_families.pdf, entitled
Maximum likelihood estimation.

Suppose you have some samples from a distribution family with unknown
parameters, and you want to estimate the maximum likelihood parmaters of the
distribution.
"""
import jax.numpy as jnp
import jax.random as jr
from tjax import print_generic

from efax import DirichletEP, DirichletNP, MaximumLikelihoodEstimator, parameter_mean

# Consider a Dirichlet distribution with a given alpha.
alpha = jnp.asarray([2.0, 3.0, 4.0])
source_distribution = DirichletNP(alpha_minus_one=alpha - 1.0)

# Let's sample from it.
n_samples = 10000
key_a = jr.key(123)
samples = source_distribution.sample(key_a, (n_samples,))

# Now, let's find the maximum likelihood Dirichlet distribution that fits it.
# First, convert the samples to their sufficient statistics.
estimator = MaximumLikelihoodEstimator.create_simple_estimator(DirichletEP)
ss = estimator.sufficient_statistics(samples)
# ss has type DirichletEP.  This is similar to the conjguate prior of the
# Dirichlet distribution.

# Take the mean over the first axis.
ss_mean = parameter_mean(ss, axis=0)  # ss_mean also has type DirichletEP.

# Convert this back to the natural parametrization.
estimated_distribution = ss_mean.to_nat()
print_generic({"estimated_distribution": estimated_distribution,
               "source_distribution": source_distribution})
# dict
# ├── estimated_distribution=DirichletNP[dataclass]
# │   └── alpha_minus_one=Jax Array (3,) float32
# │       └──  0.9797 │ 1.9539 │ 2.9763
# └── source_distribution=DirichletNP[dataclass]
#     └── alpha_minus_one=Jax Array (3,) float32
#         └──  1.0000 │ 2.0000 │ 3.0000
