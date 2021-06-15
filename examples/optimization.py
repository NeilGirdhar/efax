import jax.numpy as jnp
from jax import grad, jit, lax
from tjax import BooleanNumeric, RealNumeric

from efax import BernoulliEP, BernoulliNP


def cross_entropy_loss(p: BernoulliEP, q: BernoulliNP) -> RealNumeric:
    return p.cross_entropy(q)


gce = jit(grad(cross_entropy_loss, 1))


def body_fun(q: BernoulliNP) -> BernoulliNP:
    return BernoulliNP(q.log_odds - gce(some_p, q).log_odds * 1e-4)


def cond_fun(q: BernoulliNP) -> BooleanNumeric:
    return jnp.sum(gce(some_p, q).log_odds ** 2) > 1e-7


# some_p are expectation parameters of a Bernoulli distribution corresponding
# to probability 0.4.
some_p = BernoulliEP(jnp.array(0.4))

# some_q are natural parameters of a Bernoulli distribution corresponding to
# log-odds 0, which is probability 0.5.
some_q = BernoulliNP(jnp.array(0.0))

# Optimize the predictive distribution iteratively.
print(lax.while_loop(cond_fun, body_fun, some_q))
# Outputs the natural parameters that correspond to 0.4.

# Compare with the true value.
print(some_p.to_nat())
