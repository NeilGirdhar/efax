# Contributing

Contributions are welcome, including new features, design ideas, and new distributions.

## Adding a distribution

A new distribution usually takes about a hundred lines of code:

- Create an issue for the new distribution.
- Implement the natural and expectation parametrizations, either:
    - directly, like the Bernoulli distribution, or
    - as a transformation of an existing exponential family, like the Rayleigh distribution.
- Implement the conversion from the expectation to the natural parametrization. If it has no
  analytical solution, the `ExpToNat` mixin solves it numerically, as in the Dirichlet
  distribution.
- Add the distribution to the tests in
  [create_info](https://github.com/NeilGirdhar/efax/blob/main/tests/create_info.py).

The implementation should match the surrounding style, carry type annotations, and pass the
checks below.

## Testing and checks

Run the tests with `pytest -n auto`. To test specific distributions, pass their names as they
appear in
[create_info](https://github.com/NeilGirdhar/efax/blob/main/tests/create_info.py), for example
`pytest -n auto --distribution=Gamma`.

Clean and check the source with:

- `uv run ruff check`
- `uv run ruff format`
- `uv run ty check`
