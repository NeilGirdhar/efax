import logging
import subprocess
from typing import Any, Generator, List

import numpy as np
import pytest
from fastlog import log
from numpy.random import Generator as NumpyGenerator

from .create_info import create_infos
from .distribution_info import DistributionInfo


@pytest.fixture
def no_log() -> Generator[None, None, None]:
    log.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        log.setLevel(logging.INFO)


@pytest.fixture
def generator() -> NumpyGenerator:
    return np.random.default_rng(123)


@pytest.fixture(scope='session')
def configure_numpy() -> Generator[None, None, None]:
    line_width = [
        int(x)
        for x in subprocess.check_output(['stty',
                                          'size']).decode().split()][1]

    with np.errstate(all='raise'):
        with np.printoptions(precision=10, linewidth=line_width):
            yield


@pytest.fixture(scope='session',
                params=create_infos())
def distribution_info(request: Any) -> List[DistributionInfo]:
    return request.param
