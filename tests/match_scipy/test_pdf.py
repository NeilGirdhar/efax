from __future__ import annotations

from typing import Any

import numpy as np
import scipy.stats as ss
from jax import Array
from numpy.random import Generator
from numpy.testing import assert_allclose
from tjax import JaxComplexArray

from efax import JointDistributionN, Multidimensional, NaturalParametrization, SimpleDistribution

from ..create_info import MultivariateDiagonalNormalInfo
from ..distribution_info import DistributionInfo


def _check_observation_shape(nat_parameters: NaturalParametrization,
                            efax_x: JaxComplexArray | dict[str, Any],
                            distribution_shape: tuple[int, ...],
                            ) -> None:
    """Verify that the sufficient statistics have the right shape."""
    if isinstance(nat_parameters, JointDistributionN):
        assert isinstance(efax_x, dict)
        for name, value in nat_parameters.sub_distributions().items():
            _check_observation_shape(value, efax_x[name], distribution_shape)
        return
    assert isinstance(nat_parameters, SimpleDistribution)  # type: ignore[unreachable]
    assert isinstance(efax_x, Array)  # type: ignore[unreachable]
    dimensions = (nat_parameters.dimensions()
                  if isinstance(nat_parameters, Multidimensional)
                  else 0)
    ideal_shape = distribution_shape + nat_parameters.domain_support().shape(dimensions)
    assert efax_x.shape == ideal_shape


def test_pdf(generator: Generator, distribution_info: DistributionInfo) -> None:
    """Test that the density/mass function calculation matches scipy's."""
    distribution_shape = (10,)
    nat_parameters = distribution_info.nat_parameter_generator(generator, shape=distribution_shape)
    scipy_distribution = distribution_info.nat_to_scipy_distribution(nat_parameters)
    scipy_x = scipy_distribution.sample(rng=generator)
    efax_x = distribution_info.scipy_to_exp_family_observation(scipy_x)
    _check_observation_shape(nat_parameters, efax_x, distribution_shape)

    # Verify that the density matches scipy.
    efax_density = np.asarray(nat_parameters.pdf(efax_x), dtype=np.float64)
    if isinstance(scipy_distribution,
                  ss._distribution_infrastructure.ContinuousDistribution):  # noqa: SLF001
        scipy_density = scipy_distribution.pdf(scipy_x)  # pyright: ignore
    else:
        assert isinstance(scipy_distribution,
                          ss._distribution_infrastructure.DiscreteDistribution)  # noqa: SLF001
        scipy_density = scipy_distribution.pmf(scipy_x)  # type: ignore # pyright: ignore

    if isinstance(distribution_info, MultivariateDiagonalNormalInfo):
        atol = 1e-5
        rtol = 3e-4
    else:
        atol = 1e-5
        rtol = 1e-4
    assert_allclose(efax_density, scipy_density, rtol=rtol, atol=atol)
