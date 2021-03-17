=================================
EFAX: Exponential Families in JAX
=================================
.. image:: https://badge.fury.io/py/efax.svg
    :target: https://badge.fury.io/py/efax

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

Framework
=========
Representation
--------------
EFAX has a single base class for its objects: :python:`Parametrization` that encodes the
distribution family, and the parameters of the distribution.
Each such object has a shape, and so it can store any number of distributions.
When operating on such objects, NumPy's broadcasting rules apply.
This is unlike SciPy where each distribution is represented by a single object, and so a thousand
distributions need a thousand objects.

All parametrization objects are dataclasses using :python:`tjax.dataclass`.  These dataclasses are
a modification of Python's dataclasses to support JAX's type registration.  This allows easy marking
of static attributes.  In JAX, a static attribute is one that induces recompilation of a function
when it changes, and consequently there is more flexibility about what can be done with such an
attribute.  An example of a static attribute in EFAX is the failure number of the negative binomial
distribution.

Each non-static attribute in an EFAX distribution is marked with a support.  For example:

.. code:: python

    @dataclass
    class MultivariateNormalNP(NaturalParametrization['MultivariateNormalEP']):
        mean_times_precision: RealArray = distribution_parameter(VectorSupport())
        negative_half_precision: RealArray = distribution_parameter(SymmetricMatrixSupport())

In this case, we see that there are two natural parameters for the multivariate normal distribution.
If such an object :python:`x` has shape :python:`s`, then the shape of
:python:`x.negative_half_precision` is :python:`(*s, n, n)`.

Parametrizations
----------------
Each exponential family distribution has two special parametrizations: the natural and the
expectation parametrization.  (These are described in the overview pdf.)
Consequently, every distribution has two base classes, one inheriting from
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
Every :python:`Parametrization` has methods to flatten and unflatten the parameters into a single
array: :python:`flattened` and :python:`unflattened`.

Every :python:`NaturalParametrization` has methods:

- :python:`sufficient_statistics` to produce the sufficient statistics given an observation,
- :python:`pdf`, which is the density,
- :python:`fisher_information`, which is the Fisher information matrix, and
- :python:`entropy`, which is the Shannon entropy.


Every :python:`ExpectationParametrization` has a :python:`cross_entropy` method that has an
efficient, numerically optimized custom JAX gradient.  This is possible because the gradient of the
cross entropy is the difference of expectation parameters (when the expected carrier measure is
zero).

Numerical optimization
----------------------
Because of the nature of the log-normalizer and carrier measure, some methods for some distributions
require numerical optimization.  These are the conversion from expectation parameters to natural
ones, the entropy, and the cross entropy.

Distributions
=============
EFAX supports the following distributions:

- Bernoulli
- beta
- chi-square
- complex normal
- Dirichlet
- exponential
- gamma
- geometric
- isotropic normal (multivariate normal with isotropic variance)
- logarithmic
- multinomial
- multivariate normal
- multivariate unit normal (multivariate normal with isotropic unit variance)
- negative binomial
- normal
- Poisson
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

    from jax import grad, jit, lax
    from jax import numpy as jnp

    from efax import BernoulliEP, BernoulliNP


    def cross_entropy_loss(p, q):
        return p.cross_entropy(q)


    gce = jit(grad(cross_entropy_loss, 1))


    def body_fun(q):
        return BernoulliNP(q.log_odds - gce(some_p, q).log_odds * 1e-4)


    def cond_fun(q):
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

Contribution guidelines
=======================

Contributions are welcome!

If you want to add a new distribution, the steps are

- Create an issue for the new distribution.

- Solve for or research the equations needed to fill the blanks in the overview pdf, and put them in
  the issue.  I'll add them to the pdf for you.

- Implement the natural and expectation parametrizations.

- Add the new distribution to the tests.

Implementation should respect PEP8.
The tests can be run using :bash:`pytest .`
There are a few tools to clean and check the source:

- :bash:`isort .`

- :bash:`pylint efax`

- :bash:`flake8 efax`

- :bash:`mypy efax`
