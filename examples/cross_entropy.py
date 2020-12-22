from jax import numpy as jnp

from efax import BernoulliEP, BernoulliNP

# p is the expectation parameters of three Bernoulli distributions having probabilities 0.4, 0.5,
# and 0.6.
p = BernoulliEP(jnp.array([0.4, 0.5, 0.6]))

# q is the natural parameters of three Bernoulli distributions having log-odds 0, which is
# probability 0.5.
q = BernoulliNP(jnp.zeros(3))

print(p.cross_entropy(q))
# [0.6931472 0.6931472 0.6931472]

# q2 is natural parameters of Bernoulli distributions having a probability of 0.3.
p2 = BernoulliEP(0.3 * jnp.ones(3))
q2 = p2.to_nat()

print(p.cross_entropy(q2))
# [0.6955941  0.78032386 0.86505365]
# A Bernoulli distribution with probability 0.3 predicts a Bernoulli observation with probability
# 0.4 better than the other observations.
