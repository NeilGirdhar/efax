import logging
import subprocess

import numpy as np
import pytest
from fastlog import log


@pytest.fixture
def no_log():
    log.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        log.setLevel(logging.INFO)


@pytest.fixture
def generator():
    return np.random.default_rng(123)


@pytest.fixture(scope='session')
def configure_numpy():
    line_width = [
        int(x)
        for x in subprocess.check_output(['stty',
                                          'size']).decode().split()][1]

    with np.errstate(all='raise'):
        with np.printoptions(precision=10, linewidth=line_width):
            yield
