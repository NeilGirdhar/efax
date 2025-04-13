from __future__ import annotations

from typing import Any

import numpy as np
from numpy.linalg import det, inv
from numpy.random import Generator
from numpy.testing import assert_allclose
from scipy.special import digamma, gammaln
from tjax import JaxRealArray

from efax import (ExpectationParametrization, GammaEP, GammaNP, MultivariateNormalEP,
                  MultivariateNormalNP, NaturalParametrization, NormalEP, NormalNP)

from .create_info import GammaInfo, MultivariateNormalInfo, NormalInfo
from .distribution_info import DistributionInfo


def prelude(generator: Generator,
            distribution_info_kl: DistributionInfo[Any, Any, Any],
            distribution_name: str | None
            ) -> tuple[ExpectationParametrization[Any], NaturalParametrization[Any, Any],
                       JaxRealArray]:
    shape = (3, 2)
    distribution_info_kl.skip_if_deselected(distribution_name)
    x = distribution_info_kl.exp_parameter_generator(generator, shape=shape)
    y = distribution_info_kl.nat_parameter_generator(generator, shape=shape)
    my_kl = x.kl_divergence(y)
    return x, y, my_kl


def test_normal_kl(generator: Generator, distribution_name: str | None) -> None:
    """Test the KL divergence."""
    x, y, my_kl = prelude(generator, NormalInfo(), distribution_name)
    assert isinstance(x, NormalEP)
    assert isinstance(y, NormalNP)
    xm = x.mean
    ym = y.to_exp().mean
    xv = x.variance()
    yv = y.to_exp().variance()
    scipy_kl = 0.5 * (xv / yv + np.square(ym - xm) / yv - 1.0 + np.log(yv / xv))
    rtol = 1e-5
    assert_allclose(my_kl, scipy_kl, rtol=rtol)


def test_mvn_kl(generator: Generator, distribution_name: str | None) -> None:
    """Test the KL divergence."""
    x, y, my_kl = prelude(generator, MultivariateNormalInfo(dimensions=4), distribution_name)
    assert isinstance(x, MultivariateNormalEP)
    assert isinstance(y, MultivariateNormalNP)
    xm = x.mean
    ym = y.to_exp().mean
    xv = x.variance()
    yv = y.to_exp().variance()
    md = ym - xm
    tq = np.trace(inv(yv) @ xv, axis1=-2, axis2=-1)
    mvm = np.einsum("...i,...ij,...j->...", md, inv(yv), md)
    scipy_kl = 0.5 * (tq + mvm - x.dimensions()
                      + np.log(det(yv) / det(xv)))
    rtol = 1e-5
    assert_allclose(my_kl, scipy_kl, rtol=rtol)


def test_gamma_kl(generator: Generator, distribution_name: str | None) -> None:
    """Test the KL divergence."""
    x, y, my_kl = prelude(generator, GammaInfo(), distribution_name)
    assert isinstance(x, GammaEP)
    assert isinstance(y, GammaNP)
    xsh = x.to_nat().shape_minus_one + 1.0
    ysh = y.shape_minus_one + 1.0
    xsc = -np.reciprocal(x.to_nat().negative_rate)
    ysc = -np.reciprocal(y.negative_rate)
    scipy_kl = (((xsc - ysc) / ysc) * xsh + gammaln(ysh) - gammaln(xsh)
                + ysh * np.log(ysc) - xsh * np.log(xsc)
                + (xsh - ysh) * (np.log(xsc) + digamma(xsh)))
    rtol = 1e-5
    assert_allclose(my_kl, scipy_kl, rtol=rtol)
