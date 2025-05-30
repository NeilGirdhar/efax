from __future__ import annotations

from typing import Any

import pytest
from numpy.random import Generator
from tjax import assert_tree_allclose, print_generic, tree_allclose

from efax import parameters

from .create_info import GeneralizedDirichletInfo
from .distribution_info import DistributionInfo


def test_conversion(generator: Generator,
                    distribution_info: DistributionInfo[Any, Any, Any]
                    ) -> None:
    """Test that the conversion between the different parametrizations are consistent."""
    if isinstance(distribution_info, GeneralizedDirichletInfo):
        pytest.skip()

    n = 30
    shape = (n,)
    original_np = distribution_info.nat_parameter_generator(generator, shape=shape)
    intermediate_ep = original_np.to_exp()
    final_np = intermediate_ep.to_nat()

    # Check round trip.
    if not tree_allclose(final_np, original_np):
        for i in range(n):
            if not tree_allclose(final_np[i], original_np[i]):
                print_generic({"original": original_np[i],
                               "intermediate": intermediate_ep[i],
                               "final": final_np[i]})
                pytest.fail("Conversion failure")

    # Check fixed parameters.
    original_fixed = parameters(original_np, fixed=True)
    intermediate_fixed = parameters(intermediate_ep, fixed=True)
    final_fixed = parameters(final_np, fixed=True)
    assert_tree_allclose(original_fixed, intermediate_fixed)
    assert_tree_allclose(original_fixed, final_fixed)
