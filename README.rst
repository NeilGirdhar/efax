=================================
EFAX: Exponential Families in JAX
=================================
.. image:: https://badge.fury.io/py/efax.svg
    :target: https://badge.fury.io/py/efax

.. role:: bash(code)
    :language: bash

This library provides a set of tools for working with exponential family distributions in the differential programming library JAX.
The *exponential families* are an important class of probability distributions that include the normal, gamma, beta, exponential, Poisson, binomial, and Bernoulli distributions.
For an introduction to exponential families, see `pdf <exfam.pdf>`.

Contribution guidelines
=======================

- Conventions: PEP8.

- How to clean the source:

  - :bash:`isort -rc -y`
  - :bash:`autopep8 -r -i .`
  - :bash:`pylint *`
  - :bash:`flake8 .`
