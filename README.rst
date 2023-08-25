=================================
EFAX: Exponential Families in JAX
=================================

|pypi| |py_versions|

.. |pypi| image:: https://img.shields.io/pypi/v/efax
   :   alt: PyPI - Version

.. |py_versions| image:: https://img.shields.io/pypi/pyversions/efax
   :   alt: PyPI - Python Version

.. role:: bash(code)
    :language: bash

.. role:: python(code)
    :language: python

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
EFAX has a single base class for its objects: :python:`Parametrization` whose type encodes the
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

Every :python:`Parametrization` has methods:

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

    - with fixed unit variance
    - with arbitrary variance

  - univariate complex

    - with fixed unit variance and zero pseudo-variance
    - with arbitrary variance

  - multivariate real:

    - with fixed unit variance
    - with fixed variance
    - with isotropic variance
    - with diagonal variance
    - with arbitrary variance

  - multivariate complex:

    - with fixed unit variance and zero pseudo-variance
    - circularly symmetric

- on a finite set:

  - Bernoulli
  - multinomial

- on the nonnegative integers:

  - geometric
  - logarithmic
  - negative binomial
  - Poisson

- on the positive reals:

  - chi
  - chi-square
  - exponential
  - gamma
  - Rayleigh
  - Weibull

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

Optimization
------------
Using the cross entropy to iteratively optimize a prediction is simple:

.. code:: python

    from __future__ import annotations

    import jax.numpy as jnp
    from jax import grad, jit, lax
    from jax.tree_util import tree_map, tree_reduce
    from tjax import BooleanNumeric, RealArray, RealNumeric, print_generic

    from efax import BernoulliEP, BernoulliNP


    def cross_entropy_loss(p: BernoulliEP, q: BernoulliNP) -> RealNumeric:
        return jnp.sum(p.cross_entropy(q))


    gce = jit(grad(cross_entropy_loss, 1))


    def apply(x: RealArray, x_bar: RealArray) -> RealArray:
        return x - 1e-4 * x_bar


    def body_fun(q: BernoulliNP) -> BernoulliNP:
        q_bar = gce(some_p, q)
        return tree_map(apply, q, q_bar)


    def cond_fun(q: BernoulliNP) -> BooleanNumeric:
        q_bar = gce(some_p, q)
        total = tree_reduce(jnp.sum,
                            tree_map(lambda x: jnp.sum(jnp.square(x)), q_bar))
        return total > 1e-6


    # some_p are expectation parameters of a Bernoulli distribution corresponding
    # to probabilities 0.3, 0.4, and 0.7.
    some_p = BernoulliEP(jnp.array([0.3, 0.4, 0.7]))

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

Maximum likelihood estimation
-----------------------------
Maximum likelihood estimation is often using the conjugate prior, but this can be done using only
the expectation parametrization (which is equivalent less one parameter that represents the number
of samples).

.. code:: python

    from functools import partial

    import jax.numpy as jnp
    from jax.random import PRNGKey
    from jax.tree_util import tree_map

    from efax import DirichletNP

    # Consider a Dirichlet distribution with a given alpha.
    alpha = jnp.asarray([2.0, 3.0, 4.0])
    source_distribution = DirichletNP(alpha_minus_one=alpha - 1.0)

    # Let's sample from it.
    n_samples = 10000
    key = PRNGKey(123)
    samples = source_distribution.sample(key, (n_samples,))

    # Now, let's find the maximum likelihood Dirichlet distribution that fits it.
    # First, convert the samples to their sufficient statistics.
    ss = DirichletNP.sufficient_statistics(samples)
    # ss has type DirichletEP.  This is similar to the conjguate prior of the Dirichlet distribution.

    # Take the mean over the first axis.
    ss_mean = tree_map(partial(jnp.mean, axis=0), ss)  # ss_mean also has type DirichletEP.

    # Convert this back to the natural parametrization.
    estimated_distribution = ss_mean.to_nat()
    print(estimated_distribution.alpha_minus_one + 1.0)  # [1.9849904 3.0065458 3.963935 ]


Contribution guidelines
=======================

Contributions are welcome!

It's not hard to add a new distribution.  The steps are:

- Create an issue for the new distribution.

- Solve for or research the equations needed to fill the blanks in the overview pdf, and put them in
  the issue.  I'll add them to the pdf for you.

- Implement the natural and expectation parametrizations, either:

  - directly like in the Bernoulli distribution, or
  - as a transformation of an existing exponential family like the Rayleigh distribution.

- Implement the conversion from the expectation to the natural parametrization.  If this has no
  analytical solution, then there's a mixin that implements a numerical solution.  This can be seen
  in the Dirichlet distribution.

- Add the new distribution to the tests by adding it to `create_info <https://github.com/NeilGirdhar/efax/blob/master/tests/create_info.py>`_.

Implementation should respect PEP8.
The tests can be run using :bash:`pytest . -n auto`.  Specific distributions can be run with
:bash:`pytest . -n auto --distribution=Gamma` where the names match the class names in
`create_info <https://github.com/NeilGirdhar/efax/blob/master/tests/create_info.py>`_.

There are a few tools to clean and check the source:

- :bash:`ruff .`
- :bash:`pyright`
- :bash:`mypy`
- :bash:`isort .`
- :bash:`pylint efax tests`
