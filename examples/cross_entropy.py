from jax import numpy as jnp

from efax import Bernoulli

b = Bernoulli(shape=(3,))

# p are expectation parameters of Bernoulli distributions having probabilities
# 0.4, 0.5, and 0.6.
p = jnp.array([[0.4], [0.5], [0.6]])

# q are natural parameters of Bernoulli distributions having log-odds 0, which
# is probability 0.5.
q = jnp.zeros((3, 1))

print(b.cross_entropy(p, q))
# [0.6931472 0.6931472 0.6931472]

# q are natural parameters of Bernoulli distributions having a probability
# of 0.3.
q = b.exp_to_nat(0.3 * jnp.ones((3, 1)))

print(b.cross_entropy(p, q))
# [0.6955941  0.78032386 0.86505365]
# A Bernoulli distribution with probability 0.3 predicts a Bernoulli
# observation with probability 0.4 better than the other observations.
