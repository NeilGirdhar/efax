# Structure utilities

The structure utilities work with a distribution's structure separately from its values: its
types, parameter paths, and dimensions. They let code rebuild distributions from raw arrays,
estimate them from observations, and encode them for neural networks, without passing type
information alongside the arrays.

## Assembler, Estimator, and Flattener

Three classes capture this static metadata without requiring a live instance. Each extends the
previous one.

### Assembler

`Assembler` stores a post-order traversal of a `Distribution` tree, including its types, paths,
and dimensions, so that it can rebuild distributions from parameter data. Its main methods are:

- `assemble(params)`, which rebuilds a `Distribution` from a mapping of paths to arrays,
- `coerce_from_distribution(q)`, which reinterprets the values of `q` under this assembler's
  types,
- `domain_support()`, which lists each leaf distribution's parameter constraints,
- `generate_random(xp, rng, shape, safety)`, which draws a random distribution with valid
  parameters, and
- `to_nat()` and `to_exp()`, which return a copy whose types are all natural or all expectation
  parametrizations.

### Estimator

`Estimator` extends `Assembler` with maximum likelihood estimation. It records which parameters are
variable and which are fixed, holding constant across observations. The maximum likelihood
estimate of an exponential family is the mean of the sufficient statistics, so estimation takes
two methods:

- `sufficient_statistics(x)`, which computes the sufficient statistics of the observation `x` and
  supplies the fixed parameters, and
- `mle(x, axis=...)`, which averages the sufficient statistics over the axes along which the
  observations are stacked.

Create an estimator with `Estimator.from_type(type_p, **fixed)`, `Estimator.from_expectation(p)`,
or `Estimator.from_natural(p)`.

### Flattener

`Flattener` extends `Estimator` with encoding and decoding between a `Distribution` and an array
of shape `(*distribution.shape, k)`, so that neural networks and numerical optimizers can produce
and consume distributions. The encoded array excludes fixed parameters, and decoding reinserts
them.

- `Flattener.flatten(p, mapped_to_plane=True)` encodes `p` into a `(Flattener, array)` pair.
- `unflatten(array)` decodes the array back into a distribution.
- `final_dimension_size()` returns `k`, the size of the encoded array's last axis.

The `mapped_to_plane` flag controls whether constrained parameters, such as those on a simplex or
restricted to the positive reals, are mapped bijectively to all of ℝⁿ. Set it to true when a
neural network produces the array, so that every output decodes to a valid distribution. Set it to
false when the raw magnitudes matter, for example when differencing expectation parameters or
computing Jacobians.

## Parameter operations

These functions operate on the parameters of distributions while preserving their types. Each axis
argument indexes the distribution's shape, so vector and matrix parameters keep their own axes.

- `parameter_map(operation, x, *ys, fixed=False)` applies `operation` to corresponding parameters
  of `x` and `ys`. By default, it operates on the variable parameters and copies the fixed ones
  from `x`. With `fixed=True`, it operates on every parameter.
- `parameter_mean(x, axis=...)` averages every parameter over axes of the distribution's shape.
- `parameter_concat(xs, axis=...)` concatenates distributions along an axis of their shape.
- `parameter_dot_product(x, y)` returns the Hermitian dot product over the variable parameters,
  and `parameter_holomorphic_dot(x, y)` returns the dot product without conjugation.
- `parameters(p, fixed=..., support=...)` returns a distribution's parameters by path, optionally
  with their supports.

For example, `parameter_map(add, prior, likelihood)` combines the natural parameters of a prior
and a likelihood into a posterior, and leaves fixed parameters, such as a negative binomial
distribution's number of failures, unchanged.
