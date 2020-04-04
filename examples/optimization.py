from jax import grad, jit, lax
from jax import numpy as jnp

from efax import Bernoulli

b = Bernoulli(shape=())


def cross_entropy_loss(p, q):
    return b.cross_entropy(p, q)


gce = jit(grad(cross_entropy_loss, 1))


def body_fun(q):
    return q - gce(some_p, q) * 1e-4


def cond_fun(q):
    return jnp.sum(gce(some_p, q) ** 2) > 1e-7


# some_p are expectation parameters of a Bernoulli distribution corresponding
# to probability 0.4.
some_p = jnp.array([0.4])

# some_q are natural parameters of a Bernoulli distribution corresponding to
# log-odds 0, which is probability 0.5.
some_q = jnp.array([0.0])

# Optimize the predictive distribution iteratively.
print(lax.while_loop(cond_fun, body_fun, some_q))
# Outputs the natural parameters that correspond to 0.4.

# Compare with the true value.
print(b.exp_to_nat(some_p))
