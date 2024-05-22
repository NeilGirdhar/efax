from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
import pytest
from jax import enable_custom_prng
from jax.experimental import enable_x64
from jax.random import key as jax_key
from numpy.random import Generator as NumpyGenerator
from tjax import KeyArray

from efax import (BooleanRing, HasConjugatePrior, HasEntropyEP, HasEntropyNP,
                  HasGeneralizedConjugatePrior, IntegralRing, JointDistribution, Samplable,
                  Structure)

from .create_info import (BetaInfo, ChiSquareInfo, ComplexCircularlySymmetricNormalInfo,
                          DirichletInfo, GammaInfo, GeneralizedDirichletInfo, JointInfo,
                          create_infos)
from .distribution_info import DistributionInfo


@pytest.fixture(autouse=True)
def _jax_enable64() -> Generator[None, None, None]:
    with enable_x64():
        yield


def pytest_addoption(parser: Any) -> None:
    parser.addoption('--distribution', action='store', default=None)


@pytest.fixture()
def generator() -> NumpyGenerator:
    return np.random.default_rng(123)


@pytest.fixture()
def key() -> KeyArray:
    with enable_custom_prng():
        return jax_key(123)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "nondistribution: mark a test as not related to a particular distribution")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--distribution") is None:
        return
    skip_non_distribution = pytest.mark.skip(reason="Distribution selected")
    for item in items:
        if "nondistribution" in item.keywords:
            item.add_marker(skip_non_distribution)


_all_infos = create_infos()


@pytest.fixture()
def distribution_name(request: Any) -> None | str:
    return request.config.getoption("--distribution")


@pytest.fixture(scope='session', params=_all_infos)
def distribution_info(request: Any) -> DistributionInfo[Any, Any, Any]:
    distribution_name_option: str | None = request.config.getoption('--distribution')
    info = request.param
    assert isinstance(info, DistributionInfo)
    info.skip_if_deselected(distribution_name_option)
    return request.param


def supports(s: Structure[Any], abc: type[Any]) -> bool:
    return all(issubclass(info.type_, abc) or issubclass(info.type_, JointDistribution)
               for info in s.distributions)


def any_integral_supports(structure: Structure[Any]) -> bool:
    return any(isinstance(s.ring, BooleanRing | IntegralRing)
               for s in structure.domain_support().values())


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if 'sampling_distribution_info' in metafunc.fixturenames and 'natural' in metafunc.fixturenames:
        p = [(info, natural)
             for info in _all_infos
             for natural in (False, True)
             if supports(info.nat_structure() if natural else info.exp_structure(),
                         Samplable)]
        ids = [f"{info.name()}{'NP' if natural else 'EP'}" for info, natural in p]
        metafunc.parametrize(("sampling_distribution_info", "natural"), p,
                             ids=ids)
    if ('sampling_wc_distribution_info' in metafunc.fixturenames
            and 'natural' in metafunc.fixturenames):
        p = [(info, natural)
             for info in _all_infos
             for natural in (False, True)
             if supports(structure := (info.nat_structure() if natural else info.exp_structure()),
                         Samplable)
             if not any_integral_supports(structure)
             if not isinstance(info,
                               ComplexCircularlySymmetricNormalInfo | BetaInfo | DirichletInfo
                               | ChiSquareInfo | GammaInfo | JointInfo)]
        ids = [f"{info.name()}{'NP' if natural else 'EP'}" for info, natural in p]
        metafunc.parametrize(("sampling_wc_distribution_info", "natural"), p,
                             ids=ids)
    if 'cp_distribution_info' in metafunc.fixturenames:
        q = [info
             for info in _all_infos
             if supports(info.exp_structure(), HasConjugatePrior)]
        metafunc.parametrize("cp_distribution_info", q, ids=[info.name() for info in q])
    if 'gcp_distribution_info' in metafunc.fixturenames:
        q = [info
             for info in _all_infos
             if supports(info.exp_structure(), HasGeneralizedConjugatePrior)]
        metafunc.parametrize("gcp_distribution_info", q, ids=[info.name() for info in q])
    if 'entropy_distribution_info' in metafunc.fixturenames:
        q = [info
             for info in _all_infos
             if ((supports(info.exp_structure(), HasEntropyEP)
                  or supports(info.nat_structure(), HasEntropyNP))
                 and not isinstance(info, GeneralizedDirichletInfo))]
        metafunc.parametrize("entropy_distribution_info", q, ids=[info.name() for info in q])
