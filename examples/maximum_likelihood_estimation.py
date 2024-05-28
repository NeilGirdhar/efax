
import jax.numpy as jnp
from jax.random import key

from efax import DirichletEP, DirichletNP, MaximumLikelihoodEstimator, parameter_mean

# Consider a Dirichlet distribution with a given alpha.
alpha = jnp.asarray([2.0, 3.0, 4.0])
source_distribution = DirichletNP(alpha_minus_one=alpha - 1.0)

# Let's sample from it.
n_samples = 10000
key_a = key(123)
samples = source_distribution.sample(key_a, (n_samples,))

# Now, let's find the maximum likelihood Dirichlet distribution that fits it.
# First, convert the samples to their sufficient statistics.
estimator = MaximumLikelihoodEstimator.create_simple_estimator(DirichletEP)
ss = estimator.sufficient_statistics(samples)
# ss has type DirichletEP.  This is similar to the conjguate prior of the Dirichlet distribution.

# Take the mean over the first axis.
ss_mean = parameter_mean(ss, axis=0)  # ss_mean also has type DirichletEP.

# Convert this back to the natural parametrization.
estimated_distribution = ss_mean.to_nat()
print(estimated_distribution.alpha_minus_one + 1.0)  # [1.9849904 3.0065458 3.963935 ]  # noqa: T201
