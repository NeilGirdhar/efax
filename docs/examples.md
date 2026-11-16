# Examples

Each example below is a runnable script in the
[examples](https://github.com/NeilGirdhar/efax/tree/main/examples) directory. Together they show
how the two parametrizations make common machine learning operations short.

## Cross entropy

[`cross_entropy.py`](https://github.com/NeilGirdhar/efax/blob/main/examples/cross_entropy.py)
computes the cross entropy between observations in the expectation parametrization and predictions
in the natural parametrization, the pairing that makes cross entropy cheapest:

```python
p = BernoulliEP(jnp.asarray([0.4, 0.5, 0.6]))
q = BernoulliNP(jnp.zeros(3))
print_generic(p.cross_entropy(q))
```

It follows section 1.4.1 of the
[overview of exponential families](https://github.com/NeilGirdhar/efax/blob/main/exponential_families.pdf).

## Bayesian evidence combination

[`bayesian_evidence_combination.py`](https://github.com/NeilGirdhar/efax/blob/main/examples/bayesian_evidence_combination.py)
combines a prior and a likelihood into a posterior. Independent evidence adds in the natural
parametrization, so the posterior is a sum:

```python
prior_np = prior.to_nat()
likelihood_np = likelihood.to_nat()
posterior_np = parameter_map(add, prior_np, likelihood_np)
posterior = posterior_np.to_variance_parametrization()
```

`parameter_map` adds only the variable parameters, so a fixed parameter, such as a negative
binomial distribution's number of failures, is never summed. The example follows section 1.2.1 of
the overview.

## Optimization

[`optimization.py`](https://github.com/NeilGirdhar/efax/blob/main/examples/optimization.py) fits a
predictive distribution to a target by gradient descent on their cross entropy, inside a
`lax.while_loop`. The gradient with respect to a distribution is itself a distribution, so the
update is a `parameter_map`:

```python
gradient_cross_entropy = jit(grad(cross_entropy_loss, 1))
q_new = parameter_map(apply, q, q_bar)
```

## Maximum likelihood estimation

[`maximum_likelihood_estimation.py`](https://github.com/NeilGirdhar/efax/blob/main/examples/maximum_likelihood_estimation.py)
recovers a Dirichlet distribution from its samples. Maximum likelihood estimation often relies on a
conjugate prior, which may not be implemented for the family. Averaging the sufficient statistics
in the expectation parametrization is simpler:

```python
estimator = Estimator.from_type(DirichletEP)
ss = estimator.sufficient_statistics(samples)
ss_mean = parameter_mean(ss, axis=0)
estimated_distribution = ss_mean.to_nat()
```

`estimator.mle(samples, axis=0)` computes and averages the sufficient statistics in one call. The
example follows section 1.3.2 of the overview.
