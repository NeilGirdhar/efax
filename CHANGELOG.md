# Changelog

This changelog summarizes EFAX minor release lines inferred from version changes in
`pyproject.toml`. Patch releases are folded into their minor release line, so each section covers
changes since the previous minor release line.

## 2.2.0 - 2026-04-01

- Renamed `Structure` to `Assembler` and `MaximumLikelihoodEstimator` to `Estimator`.
- Polished the assembler, estimator, and flattener interfaces and documentation.
- Renamed the multinomial distribution to categorical.
- Added the PyPI publish workflow and switched packaging to the `uv` build backend.
- Fixed von Mises Fisher tests and refreshed ty, Ruff, README, and typing configuration.

## 2.1.0 - 2026-03-29

- Added the Wishart distribution.
- Added `ComplexVonMises`.
- Added characteristic-function support for vector- and matrix-valued sufficient statistics.
- Removed direct SciPy and TensorFlow Probability runtime dependencies in the 2.1 patch line.
- Switched characteristic-function internals from bilinear to Hermitian pairing.

## 2.0.0 - 2026-03-28

- Added the first characteristic-function implementation.
- Switched type checking from Pyright to ty.
- Updated for JAX 0.9.
- Fixed multinomial behavior and added tests.
- Modernized tooling and refreshed the exponential-families overview PDF.

## 1.23.0 - 2025-09-22

- Supported Python 3.14 and dropped Python 3.11.
- Made the test suite compatible with JAX 0.7.2.
- Fixed multivariate SciPy-distribution shape handling.
- Switched some testing and dependency maintenance to newer lock and tooling behavior.

## 1.22.0 - 2025-05-21

- Updated to JAX 0.6.1.
- Switched to `opt-einsum` where the Array API namespace lacks `einsum`.
- Switched to `tjax.stop_gradient` and `xpx.one_hot`.
- Adopted dependency groups and Lefthook for development tooling.
- Enabled additional Ruff rules and switched to `tfp-nightly`.
- Fixed import indentation, log-normal metadata, type errors, and RNG/key reuse issues.

## 1.21.0 - 2025-04-13

- Added the inverse Gaussian distribution.
- Split structure and parameter packages into smaller modules.
- Eliminated JAX initialization on import.
- Fixed scalar support-bound shapes, Hessian key reuse, and sampling behavior.
- Improved examples, docstrings, and type annotations.

## 1.20.0 - 2025-03-26

- Added the inverse gamma distribution.
- Added conjugate-prior support for log-normal and softplus-normal distributions.
- Improved transformed-distribution carrier-measure handling.
- Continued refactoring normal-family distribution modules and documentation.

## 1.19.0 - 2025-03-25

- Added the log-normal, softplus-normal, and unit-softplus-normal distributions.
- Added unit-variance log-normal support.
- Added degenerate tests and improved Gaussian reparametrization testing.
- Renamed variance-parametrization helpers for clarity.

## 1.18.0 - 2025-01-07

- Updated for SPEC 0.
- Switched more internals to Array API helper utilities.
- Added scipy-stubs support and refined developer dependency handling.
- Removed MyPy overrides and migrated more checking to the newer toolchain.
- Improved flattener interfaces and test performance in the patch line.

## 1.17.0 - 2024-10-21

- Dropped Python 3.10 and unblocked Python 3.13.
- Updated to JAX 0.4.34.
- Added Bayesian evidence-combination and conjugate-prior recovery support.
- Continued Array API migration and simplified concatenation utilities.
- Updated license metadata and pre-commit tooling.

## 1.16.1 - 2024-06-12

- Improved `JointDistribution` design and sampling behavior.
- Added automatic distribution-parameter generation.
- Made distributions tree-like.
- Reworked sufficient-statistics and flatten/unflatten behavior.
- Added complex-normal flattening and sampling support.
- Added entropy-gradient tests and switched nonlinear solving to Optimistix.
- Switched packaging experiments to uv/hatch during the patch line.

## 1.15.0 - 2023-10-09

- Supported Python 3.12 and dropped Python 3.9.
- Updated to newer JAX, Ruff, Pyright, and tjax integrations.
- Improved complex-function organization and KL-divergence typing.
- Added geometric and Rayleigh sampling support.
- Simplified `ExpToNat` and related test utilities.
- Fixed shape, gamma-generation, and type-checking issues across the patch line.

## 1.14.0 - 2023-04-24

- Dropped Python 3.8.
- Added mean-variance parametrization for the gamma distribution.
- Added Boolean and integral field support.
- Added automatic abstract-method decoration via `JaxAbstractClass`.
- Added fixed gamma generator support.
- Updated Ruff, pylint, JAX random utilities, and pyright behavior.

## 1.13.1 - 2023-03-14

- Updated to JAX types from tjax 0.28.4.
- Added `pdf` and `to_nat` to `MultivariateDiagonalNormalVP`.
- Added `log_pdf`.
- Added sampling to the negative binomial distribution.
- Added `HasEntropyEP` and `HasEntropyNP` base classes.
- Reorganized interfaces and mixins, domain-support handling, and natural-parametrization lookup.
- Added `NormalVP`, `ComplexUnitNormal`, and unit-normal support.

## 1.12.0 - 2023-02-24

- Added support for tjax 0.27 and 0.28.
- Added chi-square sampling support.
- Reordered negative-binomial arguments and fixed related type errors.
- Improved test skipping and switched more code to `tjax.jit`.
- Removed pycodestyle and adopted `typing_extensions.override`.

## 1.11.0 - 2023-02-06

- Updated to MyPy 1.0 and adopted `Self`.
- Switched to Ruff and added Ruff rule configuration.
- Added `NaturalParametrization.kl_divergence`.
- Renamed RNG parameters to key terminology.
- Fixed type errors and polished negative-binomial handling.

## 1.10.0 - 2022-12-18

- Supported Python 3.11.
- Updated for tjax 0.24.1.
- Added KL-divergence testing and optimized KL-divergence behavior.
- Fixed circularly symmetric random sampling and NumPy compatibility issues.
- Replaced more numeric aliases with array-oriented annotations.

## 1.9.0 - 2022-09-28

- Adopted `jax.Array`.
- Continued lock and annotation updates for newer dependency versions.

## 1.8.0 - 2022-09-14

- Switched to `inverse_softplus` from tjax.
- Replaced `tjax.Generator` with `jax.random.KeyArray`.
- Updated lock files and repaired annotations.

## 1.7.0 - 2022-07-01

- Added `HasGeneralizedConjugatePrior` and `Multidimensional`.
- Added generalized Dirichlet support.
- Added generalized conjugate-prior support for the multinomial distribution.
- Improved generalized Dirichlet conversion and PDF handling.
- Allowed untyped calls while waiting for JAX array types.

## 1.6.0 - 2022-02-13

- Upgraded to JAX 0.3.
- Added entropy and KL-divergence support.
- Added fixed-variance normal support.
- Adopted Python 3.10-style annotations.
- Added generalized-conjugate-prior groundwork and improved README organization.
- Fixed scalar-shaped PDF behavior and Jupyter compatibility in the patch line.

## 1.5.0 - 2021-12-09

- Supported Python 3.10.
- Migrated toward Pyright and stricter type checking.
- Updated to NumPy 1.22 and JAX 0.3 in the patch line.
- Applied the dataclass decorator to parametrizations.
- Exported public symbols more consistently.
- Fixed gradient log-normalizer tests and removed deprecated dtype helpers.

## 1.4.0 - 2021-07-14

- Updated to tjax 0.13, then through tjax 0.16 in the patch line.
- Moved pytest-xdist to development dependencies.
- Reworked Fisher-information methods.
- Exposed `ScipyComplexNormal.init_using_angle`.
- Updated TensorFlow Probability integration and related field metadata usage.
- Fixed pylint and mypy issues.

## 1.3.0 - 2021-06-23

- Updated for NumPy 1.21.
- Added complex multivariate unit normal and circularly symmetric complex normal distributions.
- Fixed sampling and complex-array unflattening bugs.
- Fixed vectorization of normal, von Mises, and complex normal distributions.
- Made flattening always flatten to real arrays.
- Enabled additional mypy warnings and improved parallel test behavior.

## 1.2.0 - 2021-02-22

- Enabled the von Mises Fisher distribution through TensorFlow Probability.
- Added sampling methods to several distributions.
- Added slicing and dimensionality support to distributions.
- Added chi, Rayleigh, Weibull, and multivariate diagonal normal distributions.
- Moved distributions under `efax.distributions`.
- Replaced axes metadata with support objects.
- Added TensorFlow Probability and nightly TensorFlow Probability integration across the patch line.
- Added parallel testing and NumPy 1.20 support.

## 1.1.0 - 2021-01-04

- Added multivariate normal and isotropic normal distributions.
- Added Fisher information and `apply_fisher_information`.
- Added the conjugate prior of `NormalUnitVariance`.
- Added flattened and unflattened methods.
- Simplified Bernoulli and multinomial distributions.
- Removed `VonMises`.
- Split exponential-family and common base-class modules.
- Replaced the SciPy optimizer with tjax and reused the `ExpToNat` mixin in additional
  distributions.

## 1.0.0 - 2020-12-26

- Re-engineered distribution objects as dataclasses.
- Added conjugate-prior distribution support.
- Added `ExpToNat` and improved conversion behavior for chi-square, logarithmic, and Dirichlet
  distributions.
- Removed `dephell`, `ipromise`, `PyTreeLike`, and `scaled_cross_entropy`.
- Added `py.typed` and expanded type annotations.
- Improved JIT behavior, including comparison/hash support and complex-normal JVP fixes.

## 0.7.0 - 2020-12-26

- Added conjugate-prior distribution support.
- Removed the `ipromise` dependency.
- Prepared the dataclass re-engineering that became the 1.0 release.

## 0.6.0 - 2020-11-12

- Added build-system metadata and updated dependencies.
- Fixed editable installs and cleaned setup configuration.
- Added `py.typed`.
- Marked the package as tested on Python 3.9 in the patch line.
- Added JIT support to `ExpToNat` and repaired JIT and complex-normal JVP behavior.

## 0.5.0 - 2020-11-12

- Made EFAX compatible with tjax 0.4.
- Simplified logarithmic and Dirichlet expectation-to-natural conversion.
- Removed unused files.

## 0.4.0 - 2020-11-12

- Switched to `tjax.custom_jvp`.
- Added chi-square distribution support in the patch line.
- Added the `ExpToNat` mixin and fixed chi-square shape behavior.

## 0.3.0 - 2020-11-12

- Adopted tjax integration and fixed the test suite around that change.
- Updated README documentation.

## 0.2.1 - 2020-11-12

- Switched to Poetry-based project management.
- Added comparison and hashing to avoid unnecessary JIT recompilation.
- Expanded type annotations, linting, mypy fixes, and tests.
- Moved tests into the project test layout and standardized line length.
