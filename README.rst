=================================
EFAX: Exponential Families in JAX
=================================
.. image:: https://badge.fury.io/py/efax.svg
    :target: https://badge.fury.io/py/efax

.. role:: bash(code)
    :language: bash

This library provides a set of tools for working with *exponential family distributions* in the differential programming library `JAX <>`.
The *exponential families* are an important class of probability distributions that include the normal, gamma, beta, exponential, Poisson, binomial, and Bernoulli distributions.
For an explaination of the fundamental ideas behind this library, see our `overview on exponential families <https://github.com/NeilGirdhar/efax/blob/master/expfam.pdf>`.

Usage
=====
In SciPy, a distribution is represented by a single object, so a thousand distributions need a thousand objects.  Each object encodes the distribution family, and the parameters of the distribution.
EFAX has a different representation.  Each `ExponentialFamily` object encodes only the distribution family for many (say, one thousand) distributions.  The parameters of the distributions are passed in to various methods on the object to evaluate various things.  For example::

.. code:: python

    from jax import numpy as jnp

    from efax import Bernoulli

    b = Bernoulli(shape=(3,))

    # p are expectation parameters of a Bernoulli distribution corresponding to
    # probabilities 0.4, 0.5, and 0.6.
    p = jnp.array([[0.4], [0.5], [0.6]])

    # q are natural parameters of a Bernoulli distribution corresponding to
    # log-odds 0, which is probability 0.5.
    q = jnp.zeros((3, 1))

    print(b.cross_entropy(p, q))
    # [0.6931472 0.6931472 0.6931472]

    # q are natural parameters of a Bernoulli distribution corresponding to
    # a probability of 0.3.
    q = b.exp_to_nat(0.3 * jnp.ones((3, 1)))

    print(b.cross_entropy(p, q))
    # [0.6955941  0.78032386 0.86505365]
    # A Bernoulli distribution with probability 0.3 predicts a Bernoulli
    # observation with probability 0.4 better than the other observations.


With exponential families, maximum likelihood estimation is just expectation over expectation parameters.  Models that combine independent predictors just sum natural parameters.  When we want to optimize such models, we just want to take the gradient of cross entropy with respect to predictions.

Thanks to JAX, any gradient of the cross entropy will automatically be as accurate and numerically stable as possible.  This is because the gradient of the cross entropy involves the gradient of the log-normalizer, which typically has a very nice form.  For example::

.. code:: python

    import jax
    from jax import lax
    from jax import numpy as jnp

    from efax import Bernoulli

    b = Bernoulli(shape=())


    def cross_entropy_loss(p, q):
        return b.cross_entropy(p, q)


    gce = jax.jit(jax.grad(cross_entropy_loss, 1))


    def body_fun(q):
        return q - gce(p, q) * 1e-4


    def cond_fun(q):
        return jnp.sum(gce(p, q) ** 2) > 1e-7


    # p are expectation parameters of a Bernoulli distribution corresponding to
    # probability 0.4.
    p = jnp.array([0.4])

    # q are natural parameters of a Bernoulli distribution corresponding to
    # log-odds 0, which is probability 0.5.
    q = jnp.array([0.0])

    # Optimize the predictive distribution iteratively.
    print(lax.while_loop(cond_fun, body_fun, q))
    # Outputs the natural parameters that correspond to 0.4.

    # Compare with the true value.
    print(b.exp_to_nat(p))

More examples to come...

Contribution guidelines
=======================

- Conventions: PEP8.

- How to clean the source:

  - :bash:`isort -rc -y`
  - :bash:`autopep8 -r -i .`
  - :bash:`pylint *`
  - :bash:`flake8 .`
