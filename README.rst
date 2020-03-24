=======
Tikzify
=======
.. image:: https://badge.fury.io/py/tikzify.svg
    :target: https://badge.fury.io/py/tikzify

.. role:: bash(code)
    :language: bash

A set of utilities for programmatically generating TikZ code.

Contribution guidelines
=======================

- Conventions

  - Naming conventions are according to PEP8.

- How to clean the source:

  - :bash:`isort -rc -y`
  - :bash:`autopep8 -r -i .`
  - :bash:`pylint *`
  - :bash:`flake8 .`

Running
=======

- This macro is helpful for running examples:

.. code-block:: bash

    function dm {
        python "$1.py" $2 && pdflatex -shell-escape $1 && open $1.pdf
    }

- The basal ganglia example can be run by doing :bash:`dm basal_ganglia` from the examples folder.  It should produce :bash:`basal_ganglia.pdf`, which shows all of the output, as well as :bash:`figures/basal_ganglia-*.pdf`, which are the individual diagrams to be included.

- A copy of the `pdf <basal_ganglia.pdf>` is provided at the top level folder.  It shows three programmatically-generated diagrams, with various sections highlighted.

Whom do I talk to?
==================

- Neil Girdhar
