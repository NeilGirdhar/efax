"""Optimization.

This example illustrates how this library fits in a typical machine learning context.  Suppose we
have an unknown target value, and a loss function based on the cross-entropy between the target
value and a predictive distribution.  We will optimize the predictive distribution by a small
fraction of its cotangent.
"""
import jax.numpy as jnp
from jax import grad, lax
from tjax import JaxBooleanArray, JaxRealArray, jit, print_generic

from efax import BernoulliEP, BernoulliNP, parameter_dot_product, parameter_map


def cross_entropy_loss(p: BernoulliEP, q: BernoulliNP) -> JaxRealArray:
    return jnp.sum(p.cross_entropy(q))


gradient_cross_entropy = jit(grad(cross_entropy_loss, 1))


def apply(x: JaxRealArray, x_bar: JaxRealArray) -> JaxRealArray:
    return x - 1e-4 * x_bar


def body_fun(q: BernoulliNP) -> BernoulliNP:
    q_bar = gradient_cross_entropy(target_distribution, q)
    return parameter_map(apply, q, q_bar)


def cond_fun(q: BernoulliNP) -> JaxBooleanArray:
    q_bar = gradient_cross_entropy(target_distribution, q)
    total = jnp.sum(parameter_dot_product(q_bar, q_bar))
    return total > 1e-6  # noqa: PLR2004


# The target_distribution is represented as the expectation parameters of a Bernoulli distribution
# corresponding to probabilities 0.3, 0.4, and 0.7.
target_distribution = BernoulliEP(jnp.asarray([0.3, 0.4, 0.7]))

# The initial predictive distribution is represented as the natural parameters of a Bernoulli
# distribution corresponding to log-odds 0, which is probability 0.5.
initial_predictive_distribution = BernoulliNP(jnp.zeros(3))

# Optimize the predictive distribution iteratively, and output the natural parameters of the
# prediction.
predictive_distribution = lax.while_loop(cond_fun, body_fun, initial_predictive_distribution)
print_generic(predictive_distribution)
# BernoulliNP
# └── log_odds=Jax Array (3,) float32
#     └──  -0.8440 │ -0.4047 │ 0.8440

# Compare the optimized predictive distribution with the target value in the same parametrization.
print_generic(target_distribution.to_nat())
# BernoulliNP
# └── log_odds=Jax Array (3,) float32
#     └──  -0.8473 │ -0.4055 │ 0.8473

# Print the optimized natural parameters as expectation parameters.
print_generic(predictive_distribution.to_exp())
# BernoulliEP
# └── probability=Jax Array (3,) float32
#     └──  0.3007 │ 0.4002 │ 0.6993
