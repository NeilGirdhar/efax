from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.linalg import det, inv
from numpy.random import Generator
from numpy.testing import assert_allclose
from scipy.special import digamma, gammaln  # pylint: disable=no-name-in-module

from .create_info import GammaInfo, MultivariateNormalInfo, NormalInfo
from .distribution_info import DistributionInfo


@pytest.mark.parametrize('distribution_info_kl', [NormalInfo(),
                                                  MultivariateNormalInfo(dimensions=4),
                                                  GammaInfo()])
def test_kl(generator: Generator, distribution_info_kl: DistributionInfo[Any, Any, Any]) -> None:
    """Test the KL divergence."""
    shape = (3, 2)
    x = distribution_info_kl.exp_parameter_generator(generator, shape=shape)
    y = distribution_info_kl.nat_parameter_generator(generator, shape=shape)
    my_kl = x.kl_divergence(y)
    if isinstance(distribution_info_kl, NormalInfo):
        xm = x.mean
        ym = y.to_exp().mean
        xv = x.variance()
        yv = y.to_exp().variance()
        scipy_kl = 0.5 * (xv / yv + np.square(ym - xm) / yv - 1.0 + np.log(yv / xv))
    elif isinstance(distribution_info_kl, MultivariateNormalInfo):
        xm = x.mean
        ym = y.to_exp().mean
        xv = x.variance()
        yv = y.to_exp().variance()
        md = ym - xm
        tq = np.trace(inv(yv) @ xv, axis1=-2, axis2=-1)
        mvm = np.einsum("...i,...ij,...j->...", md, inv(yv), md)
        scipy_kl = 0.5 * (tq + mvm - x.dimensions()
                          + np.log(det(yv) / det(xv)))
    elif isinstance(distribution_info_kl, GammaInfo):
        xsh = x.to_nat().shape_minus_one + 1.0
        ysh = y.shape_minus_one + 1.0
        xsc = -1.0 / x.to_nat().negative_rate
        ysc = -1.0 / y.negative_rate
        scipy_kl = (((xsc - ysc)/ysc) * xsh + gammaln(ysh) - gammaln(xsh)
                    + ysh * np.log(ysc) - xsh * np.log(xsc)
                    + (xsh - ysh) * (np.log(xsc) + digamma(xsh)))
    else:
        assert False
    rtol = 1e-5
    assert_allclose(my_kl, scipy_kl, rtol=rtol)
