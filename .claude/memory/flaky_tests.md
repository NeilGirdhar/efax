---
name: Flaky test investigation
description: Ongoing investigation into intermittent test failures with a consistent η+2 symptom
type: project
---

## What fails

Multiple tests fail intermittently. Originally reported: `test_jvp[UnitVarianceSoftplusNormal]`.
Also observed: `test_ep_from_cf[UnitVarianceLogNormal/LogNormal]`, `test_pdf[SoftplusNormal/LogNormal/UnitVarianceSoftplusNormal/...]`, `test_entropy_gradient[Normal/LogNormal]`.

## It is NOT xdist

Failures reproduce in a **single sequential process** (`--dist=no`). Running
`tests/test_ep_from_cf.py` alone fails ~30% of runs. xdist amplifies the rate
but is not the cause.

## Minimal reproducer

Running these four `test_ep_from_cf` parametrizations sequentially in one process
triggers the `UnitVarianceLogNormal` failure ~1/15 times:

```
Bernoulli → Exponential → InverseGaussian → UnitVarianceLogNormal [FAILS]
```

None of the three predecessors alone triggers it. Binary search was not completed
to isolate which combination is sufficient.

## The symptom is always +2

In every observed failure the wrong answer is `η + 2 ≈ 1.011` when the correct
answer is `η ≈ -0.989121`. Both the `test_jvp` failure and the `test_ep_from_cf`
failure use `mean = -0.989121` (the first value drawn from seed 123).

The `+2` is not floating-point noise — it is exact to all digits shown.

## What +2 means mathematically

In `test_ep_from_cf`, the OLS CF inversion recovers `η+2` across all 300
frequency points, meaning `Im(log φ(t)) = (η+2)·t` systematically. This implies
`shifted.log_normalizer()` (where `shifted = UnitVarianceLogNormalNP(η + i·t)`)
computes:

```
0.5·(η + i·t)² + 2·(η + i·t) + const   [wrong]
```

instead of:

```
0.5·(η + i·t)² + const                  [correct]
```

i.e., the log-normalizer has a spurious `+2·mean` linear term.

In `test_jvp`, the custom JVP returns `parameter_dot_product(q_dot, p) = η+2`
instead of `η`, which implies `p = q.to_exp()` has `mean = η+2` instead of `η`.

Both symptoms point to the same underlying corruption: something makes the
distribution behave as if its mean is `η+2`.

## Architecture context

- `_log_normalizer_jvp` is **one Python function** registered as the JVP rule for
  every distribution's `log_normalizer`.
- All `TransformedNaturalParametrization` subclasses (`UnitVarianceLogNormalNP`,
  `UnitVarianceSoftplusNormalNP`, `LogNormalNP`, `SoftplusNormalNP`, …) share
  **one `custom_jvp_method` object** for `log_normalizer` (set on
  `TransformedNaturalParametrization` by `JaxAbstractClass.__init_subclass__`).
- `UnitVarianceLogNormalNP.carrier_measure(x) = -0.5·log(x)² - log(x)`.
  The `-log(x)` is a Jacobian correction term.
- `UnitVarianceLogNormalEP.expected_carrier_measure() = -mean - 0.5·(mean²+1)`.
  Contains a `-mean` term.

## What has been ruled out

- xdist / parallelism (fails without it)
- Floating-point rounding (difference is exactly 2.0)
- Random seed non-determinism (generator freshly seeded with 123 per test)
- JIT cache key collisions between distinct distribution types (pytree treedefs
  are distinct per class)
- Correctness of the JVP formula in isolation (tested; always gives correct `η`)

## Root cause: unknown

The `+2` is specific and reproducible, but the exact path by which JAX computes
the wrong value has not been traced. Best hypothesis: something in how the shared
`custom_jvp_method` object or `_log_normalizer_jvp` function interacts when
multiple distributions' JIT traces are active in the same process within XLA's
multithreaded execution.
