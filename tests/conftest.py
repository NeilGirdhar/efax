from __future__ import annotations

import subprocess
from collections.abc import Generator
from typing import Any

import numpy as np
import pytest
from jax.experimental import enable_x64
from jax.random import KeyArray, PRNGKey
from numpy.random import Generator as NumpyGenerator

from efax import HasConjugatePrior, HasGeneralizedConjugatePrior, Samplable

from .create_info import create_infos
from .distribution_info import DistributionInfo


@pytest.fixture(autouse=True)
def _jax_enable64() -> Generator[None, None, None]:
    with enable_x64():
        yield


def pytest_addoption(parser: Any) -> None:
    parser.addoption('--distribution', action='store', default=None)


@pytest.fixture
def generator() -> NumpyGenerator:
    return np.random.default_rng(123)


@pytest.fixture
def rng() -> KeyArray:
    return PRNGKey(123)


@pytest.fixture(scope='session')
def configure_numpy() -> Generator[None, None, None]:
    line_width = [int(x)
                  for x in subprocess.check_output(['stty', 'size']).decode().split()][1]

    with np.errstate(all='raise'):
        with np.printoptions(precision=10, linewidth=line_width):
            yield


_all_infos = create_infos()


@pytest.fixture(scope='session', params=_all_infos)
def distribution_info(request: Any  # pylint: disable=inconsistent-return-statements
                      ) -> DistributionInfo[Any, Any, Any] | None:
    distribution_name: str | None = request.config.getoption('--distribution')
    info_name = type(request.param).__name__.removesuffix('Info')
    if distribution_name is None or info_name == distribution_name:
        return request.param
    pytest.skip(f"Deselected {info_name}")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if 'sampling_distribution_info' in metafunc.fixturenames and 'natural' in metafunc.fixturenames:
        p = [(info, natural)
             for info in _all_infos
             for natural in [False, True]
             if issubclass(info.nat_class() if natural else info.exp_class(), Samplable)]
        metafunc.parametrize("sampling_distribution_info, natural", p)
    if 'cp_distribution_info' in metafunc.fixturenames:
        q = [info
             for info in _all_infos
             if issubclass(info.exp_class(), HasConjugatePrior)]
        metafunc.parametrize("cp_distribution_info", q)
    if 'gcp_distribution_info' in metafunc.fixturenames:
        q = [info
             for info in _all_infos
             if issubclass(info.exp_class(), HasGeneralizedConjugatePrior)]
        metafunc.parametrize("gcp_distribution_info", q)
