"""Cross-entropy.

This example is based on section 1.4.1 from expfam.pdf, entitled Information theoretic statistics.
"""
import jax.numpy as jnp
from tjax import print_generic

from efax import BernoulliEP, BernoulliNP

# p is the expectation parameters of three Bernoulli distributions having probabilities 0.4, 0.5,
# and 0.6.
p = BernoulliEP(jnp.asarray([0.4, 0.5, 0.6]))

# q is the natural parameters of three Bernoulli distributions having log-odds 0, which is
# probability 0.5.
q = BernoulliNP(jnp.zeros(3))

print_generic(p.cross_entropy(q))
# Jax Array (3,) float32
# └──  0.6931 │ 0.6931 │ 0.6931

# q2 is natural parameters of Bernoulli distributions having a probability of 0.3.
p2 = BernoulliEP(0.3 * jnp.ones(3))
q2 = p2.to_nat()

print_generic(p.cross_entropy(q2))
# Jax Array (3,) float32
# └──  0.6956 │ 0.7803 │ 0.8651
# A Bernoulli distribution with probability 0.3 predicts a Bernoulli observation with probability
# 0.4 better than the other observations.
