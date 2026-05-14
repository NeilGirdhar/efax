"""Shared test shapes.

JAX's jit cache is keyed by (function, abstract input shape, dtype).  When two tests use
different shapes for the same conceptual purpose (e.g. ``(3, 4)`` in one test and ``(4, 3)``
in another), the same method gets compiled twice per distribution class.  Reusing one named
shape across all tests with the same intent lets each method compile once per distribution
and then hit cache for the rest of the suite.

When picking a shape for a new test, prefer one of the constants below.  Pick the smallest
shape that is adequate; use a larger one only when a statistical requirement demands it
(MLE convergence, finite-difference gradient noise, etc.).

All shapes here are 2-D so tests exercise broadcasting in more than one axis.  Past 1-D
shapes have been eliminated — every test in the suite works with a 2-D distribution batch.
"""

from tjax import Shape

# Distribution-parameter batch shapes ---------------------------------------------------

# Smallest 2-D distribution batch (4 distributions).  Used when the test does heavy
# per-element work — autodiff over keys, large observation tensors stacked behind the
# batch, etc. — and a larger batch would multiply the cost more than the cache savings
# from sharing with DIST_SHAPE_MEDIUM would recoup.
# Used by: test_sampling (every test), test_pdf, test_reparametrization_trick.
DIST_SHAPE_SMALL: Shape = (2, 2)

# Default 2-D distribution batch (12 distributions).  The right choice for most tests:
# exercises broadcasting, has enough random parameter draws to surface bugs, and shares
# the JIT cache across the largest group of tests.
# Used by: test_shapes, test_flatten, test_kl, test_fisher_information_is_convex,
# test_conjugate_prior, test_from_conjugate_prior, test_generalized_conjugate_prior,
# test_conversion, test_hessian.
DIST_SHAPE_MEDIUM: Shape = (3, 4)

# Larger 2-D distribution batch (91 distributions).  Used when statistical noise matters
# — entropy tests run jax.test_util.check_grads, whose finite-difference gradient
# comparison needs more elements to converge to the analytic gradient.
# Used by: test_entropy_gradient, match_scipy/test_entropy.
DIST_SHAPE_LARGE: Shape = (7, 13)

# Observation-count shapes --------------------------------------------------------------

# Small sample_shape (128 observations).  Used when the test only checks tensor shapes
# / dispatch and doesn't need statistical convergence.  Kept small for speed.
# Used by: test_multidimensional_sample_shape_matches_dimensions,
# test_sufficient_statistics_consistent_shapes.
SAMPLE_SHAPE_SMALL: Shape = (8, 16)

# Large sample_shape (65 536 observations).  Used when the test needs the MLE of the
# sufficient statistics to actually converge to the true expectation parameters within
# rtol ≈ 4e-2.
# Used by: test_sampling_and_estimation, test_maximum_likelihood_estimation.
SAMPLE_SHAPE_LARGE: Shape = (1024, 64)
