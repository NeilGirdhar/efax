import subprocess
from typing import Any, Generator, Optional

import numpy as np
import pytest
from jax.experimental import enable_x64
from numpy.random import Generator as NumpyGenerator
from tjax import Generator as TjaxGenerator

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
def rng() -> TjaxGenerator:
    return TjaxGenerator.from_seed(123)


@pytest.fixture(scope='session')
def configure_numpy() -> Generator[None, None, None]:
    line_width = [int(x)
                  for x in subprocess.check_output(['stty', 'size']).decode().split()][1]

    with np.errstate(all='raise'):
        with np.printoptions(precision=10, linewidth=line_width):
            yield


@pytest.fixture(scope='session', params=create_infos())
def distribution_info(request: Any  # pylint: disable=inconsistent-return-statements
                      ) -> Optional[DistributionInfo[Any, Any, Any]]:
    distribution_name: Optional[str] = request.config.getoption('--distribution')
    info_name = type(request.param).__name__[:-4]
    if distribution_name is None or info_name == distribution_name:
        return request.param
    pytest.skip(f"Deselected {info_name}")
