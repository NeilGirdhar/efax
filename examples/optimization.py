from __future__ import annotations

import jax.numpy as jnp
from jax import grad, lax
from tjax import JaxBooleanArray, JaxRealArray, jit, print_generic

from efax import BernoulliEP, BernoulliNP, parameter_dot_product, parameter_map


def cross_entropy_loss(p: BernoulliEP, q: BernoulliNP) -> JaxRealArray:
    return jnp.sum(p.cross_entropy(q))


gce = jit(grad(cross_entropy_loss, 1))


def apply(x: JaxRealArray, x_bar: JaxRealArray) -> JaxRealArray:
    return x - 1e-4 * x_bar


def body_fun(q: BernoulliNP) -> BernoulliNP:
    q_bar = gce(some_p, q)
    return parameter_map(apply, q, q_bar)


def cond_fun(q: BernoulliNP) -> JaxBooleanArray:
    q_bar = gce(some_p, q)
    total = jnp.sum(parameter_dot_product(q_bar, q_bar))
    return total > 1e-6  # noqa: PLR2004


# some_p are expectation parameters of a Bernoulli distribution corresponding
# to probabilities 0.3, 0.4, and 0.7.
some_p = BernoulliEP(jnp.asarray([0.3, 0.4, 0.7]))

# some_q are natural parameters of a Bernoulli distribution corresponding to
# log-odds 0, which is probability 0.5.
some_q = BernoulliNP(jnp.zeros(3))

# Optimize the predictive distribution iteratively, and output the natural parameters of the
# prediction.
optimized_q = lax.while_loop(cond_fun, body_fun, some_q)
print_generic(optimized_q)
# BernoulliNP
# └── log_odds=Jax Array (3,) float32
#     └──  -0.8440 │ -0.4047 │ 0.8440

# Compare with the true value.
print_generic(some_p.to_nat())
# BernoulliNP
# └── log_odds=Jax Array (3,) float32
#     └──  -0.8473 │ -0.4055 │ 0.8473

# Print optimized natural parameters as expectation parameters.
print_generic(optimized_q.to_exp())
# BernoulliEP
# └── probability=Jax Array (3,) float32
#     └──  0.3007 │ 0.4002 │ 0.6993
