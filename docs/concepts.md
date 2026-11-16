# Concepts

EFAX represents each exponential family as a pair of parametrization classes that share one
interface. This page describes how those objects store distributions, how the two parametrizations
relate, and which methods every object provides.

## Representation

EFAX has a single base class, `Distribution`, whose subclasses encode the distribution family and
its parametrization.

Each object has a shape, so it can store any number of distributions, and operations on it are
vectorized. SciPy instead represents each distribution by its own object, so a thousand
distributions need a thousand objects and a thousand function calls.

Every object is a dataclass built with `tjax.dataclass`, which registers it as a JAX pytree. Each
field stores one parameter over a specified support. For example:

```python
@dataclass
class MultivariateNormalNP(NaturalParametrization["MultivariateNormalEP"]):
    mean_times_precision: RealArray = distribution_parameter(VectorSupport())
    negative_half_precision: RealArray = distribution_parameter(SymmetricMatrixSupport())
```

The multivariate normal distribution has two natural parameters. If an object `x` of this type
has shape `s`, then `x.mean_times_precision` has shape `(*s, n)` and `x.negative_half_precision`
has shape `(*s, n, n)`.

Some parameters are fixed: they are constant within the exponential family rather than
parameters of it. For example, the number of failures of the negative binomial distribution is
fixed. Operations that combine distributions, such as `parameter_map`, operate on the variable
parameters and carry the fixed ones through.

## Parametrizations

Each exponential family has two special parametrizations, the natural and the expectation
parametrization, which the
[overview of exponential families](https://github.com/NeilGirdhar/efax/blob/main/exponential_families.pdf)
describes. Every family therefore has at least two classes: one inheriting from
`NaturalParametrization` and one from `ExpectationParametrization`.

- The natural parametrization combines and scales independent predictive evidence. In it, these
  operations are addition and scaling.
- The expectation parametrization combines independent observations into the maximum likelihood
  distribution that could have produced them. In it, this combination is an average.

`NaturalParametrization.to_exp` and `ExpectationParametrization.to_nat` convert between the two.

Some families also provide convenience parametrizations for constructing and interpreting
distributions. They are not exponential family parametrizations. For example, the normal
distribution offers `NormalVP` (variance), `NormalDP` (deviation), and `NormalPP` (precision), and
the multivariate normal distribution offers `MultivariateNormalVP` and
`MultivariateDiagonalNormalVP`.

## Methods

EFAX provides the methods most used in machine learning.

Every `Distribution` has:

- `shape` and `ndim`, which support broadcasting, and
- indexing with `[]`, which slices all parameter arrays at once.

Every `NaturalParametrization` has:

- `to_exp`, which converts it to expectation parameters,
- `sufficient_statistics`, which produces the sufficient statistics of an observation for maximum
  likelihood estimation,
- `log_normalizer`, the log partition function,
- `carrier_measure`, the base measure,
- `pdf` and `log_pdf`, the density or mass function and its logarithm,
- `scale_evidence`, which multiplies the natural parameters by a scale to weaken or strengthen the
  distribution's evidence,
- `fisher_information_diagonal` and `fisher_information_trace`, the diagonal and trace of the
  Fisher information matrix, stored as distribution objects,
- `apply_fisher_information`, which applies the Fisher information matrix to a vector of
  expectation parameters in a single VJP pass,
- `jeffreys_prior_density`, the square root of the determinant of the Fisher information,
- `characteristic_function`, the characteristic function of the sufficient statistics, computed by
  analytic continuation of the log-normalizer, and
- `kl_divergence`, the KL divergence.

Every `ExpectationParametrization` has:

- `to_nat`, which converts it to natural parameters, and
- `kl_divergence`, the KL divergence.

The function `expectation_parameters_from_characteristic_function(t, cf_values)` estimates
expectation parameters from evaluations of the characteristic function by ordinary least squares.
It is exact for the normal distribution, whose log characteristic function is linear in the
frequencies, and a first-order approximation for other families.

### Interfaces

Some parametrizations implement these interfaces:

- `HasConjugatePrior` produces and recovers the conjugate prior.
- `HasGeneralizedConjugatePrior` adds a pseudo-observation count for each dimension.
- `Multidimensional` distributions have an integer number of `dimensions`.
- `Samplable` distributions support sampling.

These public mixins provide entropy:

- `HasEntropy` provides an `entropy` method.
- `HasEntropyEP` is an expectation parametrization with analytically tractable `entropy` and
  `cross_entropy`.
- `HasEntropyNP` is a natural parametrization with analytically tractable entropy, computed
  through its expectation parametrization.

These private mixins help implement families:

- `ExpToNat` converts expectation parameters to natural parameters when no analytical solution
  exists. It inverts the gradient of the log-normalizer by Newton's method.
- `TransformedNaturalParametrization` defines a natural parametrization by relating it to an
  existing one, and `TransformedExpectationParametrization` does the same for expectation
  parametrizations.

## Joint distributions

`JointDistribution`, `JointDistributionE`, and `JointDistributionN` combine independent
distributions into a single object. `JointDistributionE` holds expectation parametrizations and
implements `HasEntropyEP`, and `JointDistributionN` holds natural parametrizations. They convert
with `to_nat` and `to_exp` like simple distributions.
