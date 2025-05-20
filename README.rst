.. role:: bash(code)
    :language: bash

.. role:: python(code)
    :language: python

.. image:: https://img.shields.io/pypi/v/efax
   :target: https://pypi.org/project/efax/
   :alt: PyPI - Version
   :align: center
.. image:: https://img.shields.io/badge/version_scheme-EffVer-0097a7
   :alt: EffVer Versioning
   :target: https://jacobtomlinson.dev/effver
.. image:: https://img.shields.io/badge/SPEC-0-blue
   :target: https://scientific-python.org/specs/spec-0000/
   :alt: SPEC-0
   :align: center
.. image:: https://img.shields.io/pypi/pyversions/efax
   :alt: PyPI - Python Version
   :align: center

=================================
EFAX: Exponential Families in JAX
=================================

This library provides a set of tools for working with *exponential family distributions* in the
differential programming library `JAX <https://github.com/google/jax/>`_.

The *exponential families* are an important class of probability distributions that include the
normal, gamma, beta, exponential, Poisson, binomial, and Bernoulli distributions.
For an explanation of the fundamental ideas behind this library, see our `overview on exponential
families <https://github.com/NeilGirdhar/efax/blob/master/expfam.pdf>`_.

The main motivation for using EFAX over a library like tensorflow-probability or the basic functions
in JAX is that EFAX provides the two most important parametrizations for each exponential family—the
natural and expectation parametrizations—and a uniform interface to efficient implementations of the
main functions used in machine learning.  An example of why this matters is that the most efficient
way to implement cross entropy between X and Y relies on X being in the expectation parametrization
and Y in the natural parametrization.

----

.. contents::

----

Framework
=========
Representation
--------------
EFAX has a single base class for its objects: :python:`Distribution` whose type encodes the
distribution family.

Each parametrization object has a shape, and so it can store any number of distributions.
Operations on these objects are vectorized.
This is unlike SciPy where each distribution is represented by a single object, and so a thousand
distributions need a thousand objects, and corresponding calls to functions that operate on them.

All parametrization objects are dataclasses using :python:`tjax.dataclass`.  These dataclasses are
a modification of Python's dataclasses to support JAX's “PyTree” type registration.

Each of the fields of a parametrization object stores a parameter over a specified support.
Some parameters are marked as “fixed”, which means that they are fixed with respect to the
exponential family.  An example of a fixed parameter is the failure number of the negative binomial
distribution.

For example:

.. code:: python

    @dataclass
    class MultivariateNormalNP(NaturalParametrization['MultivariateNormalEP']):
        mean_times_precision: RealArray = distribution_parameter(VectorSupport())
        negative_half_precision: RealArray = distribution_parameter(SymmetricMatrixSupport())

In this case, we see that there are two natural parameters for the multivariate normal distribution.
Objects of this type can hold any number of distributions:  if such an object :python:`x` has shape
:python:`s`, then the shape of
:python:`x.mean_times_precision` is :python:`(*s, n)` and the shape of
:python:`x.negative_half_precision` is :python:`(*s, n, n)`.

Parametrizations
----------------
Each exponential family distribution has two special parametrizations: the natural and the
expectation parametrization.  (These are described in the overview pdf.)
Consequently, every distribution has at least two base classes, one inheriting from
:python:`NaturalParametrization` and one from :python:`ExpectationParametrization`.

The motivation for the natural parametrization is combining and scaling independent predictive
evidence.  In the natural parametrization, these operations correspond to scaling and addition.

The motivation for the expectation parametrization is combining independent observations into the
maximum likelihood distribution that could have produced them.  In the expectation parametrization,
this is an expected value.

EFAX provides conversions between the two parametrizations through the
:python:`NaturalParametrization.to_exp` and :python:`ExpectationParametrization.to_nat` methods.

Important methods
-----------------
EFAX aims to provide the main methods used in machine learning.

Every :python:`Distribution` has methods:

- :python:`flattened` and :python:`unflattened` to flatten and unflatten the parameters into a
  single array.  Typically, array-valued signals in a machine learning model would be unflattened
  into a distribution object, operated on, and then flattened before being sent back to the model.
  Flattening is careful with distributions with symmetric (or Hermitian) matrix-valued parameters.
  It only stores the upper triangular elements.  And,
- :python:`shape`, which supports broadcasting.

Every :python:`NaturalParametrization` has methods:

- :python:`to_exp` to convert itself to expectation parameters.
- :python:`sufficient_statistics` to produce the sufficient statistics given an observation (used in
  maximum likelihood estimation),
- :python:`pdf` and :python:`log_pdf`, which is the density or mass function and its logarithm,
- :python:`fisher_information`, which is the Fisher information matrix, and
- :python:`kl_divergence`, which is the KL divergence.

Every :python:`ExpectationParametrization` has methods:

- :python:`to_nat` to convert itself to natural parameters, and
- :python:`kl_divergence`, which is the KL divergence.

Some parametrizations inherit from these interfaces:

- :python:`HasConjugatePrior` can produce the conjugate prior,
- :python:`HasGeneralizedConjugatePrior` can produce a generalization of the conjugate prior,
- :python:`Multidimensional` distributions have a integer number of `dimensions`, and
- :python:`Samplable` distributions support sampling.

Some parametrizations inherit from these public mixins:

- :python:`HasEntropyEP` is an expectation parametrization with an entropy and cross entropy, and
- :python:`HasEntropyNP` is a natural parametrization with an entropy,  (The cross entropy is not
  efficient.)

Some parametrizations inherit from these private mixins:

- :python:`ExpToNat` implements the conversion from expectation to natural parameters when no
  analytical solution is possible.  It uses Newton's method with a Jacobian to invert the gradient
  log-normalizer.
- :python:`TransformedNaturalParametrization` produces a natural parametrization by relating it to
  an existing natural parametrization.  And similarly for
  :python:`TransformedExpectationParametrization`.

Distributions
=============
EFAX supports the following distributions:

- normal:

  - univariate real:

    - with unit variance
    - with arbitrary parameters

  - univariate complex

    - with unit variance and zero pseudo-variance
    - with arbitrary parameters

  - multivariate real:

    - with unit variance
    - with fixed variance
    - with isotropic variance
    - with diagonal variance
    - with arbitrary parameters

  - multivariate complex:

    - with unit variance and zero pseudo-variance
    - circularly symmetric

  - softplus-transformed:

    - with unit variance
    - with arbitrary parameters

  - log-normal (exponential-transformed):

    - with unit variance
    - with arbitrary parameters

- on a finite set:

  - Bernoulli
  - multinomial

- on the nonnegative integers:

  - geometric
  - logarithmic
  - negative binomial
  - Poisson

- on the positive reals:

  - Rayleigh
  - Weibull
  - chi
  - chi-square
  - exponential
  - gamma
  - inverse Gaussian
  - inverse gamma

- on the simplex:

  - beta
  - Dirichlet
  - generalized Dirichlet

- on the n-sphere:

  - von Mises-Fisher

Usage
=====
Basic usage
-----------
A basic use of the two parametrizations:

.. code:: python

    """Cross-entropy.

    This example is based on section 1.4.1 from expfam.pdf, entitled Information
    theoretic statistics.
    """
    import jax.numpy as jnp
    from tjax import print_generic

    from efax import BernoulliEP, BernoulliNP

    # p is the expectation parameters of three Bernoulli distributions having
    # probabilities 0.4, 0.5, and 0.6.
    p = BernoulliEP(jnp.asarray([0.4, 0.5, 0.6]))

    # q is the natural parameters of three Bernoulli distributions having log-odds
    # 0, which is probability 0.5.
    q = BernoulliNP(jnp.zeros(3))

    print_generic(p.cross_entropy(q))
    # Jax Array (3,) float32
    # └──  0.6931 │ 0.6931 │ 0.6931

    # q2 is natural parameters of Bernoulli distributions having a probability of
    # 0.3.
    p2 = BernoulliEP(0.3 * jnp.ones(3))
    q2 = p2.to_nat()

    # A Bernoulli distribution with probability 0.3 predicts a Bernoulli observation
    # with probability 0.4 better than the other observations.
    print_generic(p.cross_entropy(q2))
    # Jax Array (3,) float32
    # └──  0.6956 │ 0.7803 │ 0.8651

Evidence combination:

.. code:: python

    """Bayesian evidence combination.

    This example is based on section 1.2.1 from expfam.pdf, entitled Bayesian
    evidence combination.

    Suppose you have a prior, and a set of likelihoods, and you want to combine all
    of the evidence into one distribution.
    """
    from operator import add

    import jax.numpy as jnp
    from tjax import print_generic

    from efax import MultivariateDiagonalNormalVP, parameter_map

    prior = MultivariateDiagonalNormalVP(mean=jnp.zeros(2),
                                         variance=10 * jnp.ones(2))
    likelihood = MultivariateDiagonalNormalVP(mean=jnp.asarray([1.1, -2.2]),
                                              variance=jnp.asarray([3.0, 1.0]))

    # Convert to the natural parametrization.
    prior_np = prior.to_nat()
    likelihood_np = likelihood.to_nat()

    # Sum.  We use parameter_map to ensure that we don't accidentally add "fixed"
    # parameters, e.g., the failure count of a negative binomial distribution.
    posterior_np = parameter_map(add, prior_np, likelihood_np)

    # Convert to the source parametrization.
    posterior = posterior_np.to_variance_parametrization()
    print_generic({"prior": prior,
                   "likelihood": likelihood,
                   "posterior": posterior})
    # dict
    # ├── likelihood=MultivariateDiagonalNormalVP[dataclass]
    # │   ├── mean=Jax Array (2,) float32
    # │   │   └──  1.1000 │ -2.2000
    # │   └── variance=Jax Array (2,) float32
    # │       └──  3.0000 │ 1.0000
    # ├── posterior=MultivariateDiagonalNormalVP[dataclass]
    # │   ├── mean=Jax Array (2,) float32
    # │   │   └──  0.8462 │ -2.0000
    # │   └── variance=Jax Array (2,) float32
    # │       └──  2.3077 │ 0.9091
    # └── prior=MultivariateDiagonalNormalVP[dataclass]
    #     ├── mean=Jax Array (2,) float32
    #     │   └──  0.0000 │ 0.0000
    #     └── variance=Jax Array (2,) float32
    #         └──  10.0000 │ 10.0000

Optimization
------------
Using the cross entropy to iteratively optimize a prediction is simple:

.. code:: python

    """Optimization.

    This example illustrates how this library fits in a typical machine learning
    context.  Suppose we have an unknown target value, and a loss function based on
    the cross-entropy between the target value and a predictive distribution.  We
    will optimize the predictive distribution by a small fraction of its cotangent.
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


    # The target_distribution is represented as the expectation parameters of a
    # Bernoulli distribution corresponding to probabilities 0.3, 0.4, and 0.7.
    target_distribution = BernoulliEP(jnp.asarray([0.3, 0.4, 0.7]))

    # The initial predictive distribution is represented as the natural parameters
    # of a Bernoulli distribution corresponding to log-odds 0, which is probability
    # 0.5.
    initial_predictive_distribution = BernoulliNP(jnp.zeros(3))

    # Optimize the predictive distribution iteratively.
    predictive_distribution = lax.while_loop(cond_fun, body_fun,
                                             initial_predictive_distribution)

    # Compare the optimized predictive distribution with the target value in the
    # same natural parametrization.
    print_generic({"predictive_distribution": predictive_distribution,
                   "target_distribution": target_distribution.to_nat()})
    # dict
    # ├── predictive_distribution=BernoulliNP[dataclass]
    # │   └── log_odds=Jax Array (3,) float32
    # │       └──  -0.8440 │ -0.4047 │ 0.8440
    # └── target_distribution=BernoulliNP[dataclass]
    #     └── log_odds=Jax Array (3,) float32
    #         └──  -0.8473 │ -0.4055 │ 0.8473

    # Do the same in the expectation parametrization.
    print_generic({"predictive_distribution": predictive_distribution.to_exp(),
                   "target_distribution": target_distribution})
    # dict
    # ├── predictive_distribution=BernoulliEP[dataclass]
    # │   └── probability=Jax Array (3,) float32
    # │       └──  0.3007 │ 0.4002 │ 0.6993
    # └── target_distribution=BernoulliEP[dataclass]
    #     └── probability=Jax Array (3,) float32
    #         └──  0.3000 │ 0.4000 │ 0.7000

Maximum likelihood estimation
-----------------------------
Maximum likelihood estimation often uses the conjugate prior, which can require exotic conjugate
prior distributions to have been implemented.  It is simpler to use the expectation parametrization
instead.

.. code:: python

    """Maximum likelihood estimation.

    This example is based on section 1.3.2 from expfam.pdf, entitled Maximum
    likelihood estimation.

    Suppose you have some samples from a distribution family with unknown
    parameters, and you want to estimate the maximum likelihood parmaters of the
    distribution.
    """
    import jax.numpy as jnp
    import jax.random as jr
    from tjax import print_generic

    from efax import DirichletEP, DirichletNP, MaximumLikelihoodEstimator, parameter_mean

    # Consider a Dirichlet distribution with a given alpha.
    alpha = jnp.asarray([2.0, 3.0, 4.0])
    source_distribution = DirichletNP(alpha_minus_one=alpha - 1.0)

    # Let's sample from it.
    n_samples = 10000
    key_a = jr.key(123)
    samples = source_distribution.sample(key_a, (n_samples,))

    # Now, let's find the maximum likelihood Dirichlet distribution that fits it.
    # First, convert the samples to their sufficient statistics.
    estimator = MaximumLikelihoodEstimator.create_simple_estimator(DirichletEP)
    ss = estimator.sufficient_statistics(samples)
    # ss has type DirichletEP.  This is similar to the conjguate prior of the
    # Dirichlet distribution.

    # Take the mean over the first axis.
    ss_mean = parameter_mean(ss, axis=0)  # ss_mean also has type DirichletEP.

    # Convert this back to the natural parametrization.
    estimated_distribution = ss_mean.to_nat()
    print_generic({"estimated_distribution": estimated_distribution,
                   "source_distribution": source_distribution})
    # dict
    # ├── estimated_distribution=DirichletNP[dataclass]
    # │   └── alpha_minus_one=Jax Array (3,) float32
    # │       └──  0.9797 │ 1.9539 │ 2.9763
    # └── source_distribution=DirichletNP[dataclass]
    #     └── alpha_minus_one=Jax Array (3,) float32
    #         └──  1.0000 │ 2.0000 │ 3.0000

Contribution guidelines
=======================

Contributions are welcome! I'm open to both new features, design ideas, and new distributions.

It's not hard to add a new distribution.  It's usually around only one hundred lines of code. The
steps are:

- Create an issue for the new distribution.

- Implement the natural and expectation parametrizations, either:

  - directly like in the Bernoulli distribution, or
  - as a transformation of an existing exponential family like the Rayleigh distribution.

- Implement the conversion from the expectation to the natural parametrization.  If this has no
  analytical solution, then there's a mixin that implements a numerical solution.  This can be seen
  in the Dirichlet distribution.

- Add the new distribution to the tests by adding it to `create_info <https://github.com/NeilGirdhar/efax/blob/master/tests/create_info.py>`_.

The implementation should be consistent with the surrounding style, be type annotated, and pass the
linters below.

The tests can be run using :bash:`pytest -n auto`.  Specific distributions can be run with
:bash:`pytest -n auto --distribution=Gamma` where the names match the class names in
`create_info <https://github.com/NeilGirdhar/efax/blob/master/tests/create_info.py>`_.

There are a few tools to clean and check the source:

- :bash:`ruff check`
- :bash:`pyright`
- :bash:`mypy`
- :bash:`isort .`
- :bash:`pylint efax tests`
