from typing import Any, List

import pytest

from .create_info import create_infos
from .distribution_info import DistributionInfo


@pytest.fixture(scope='session',
                params=create_infos())
def distribution_info(request: Any) -> List[DistributionInfo]:
    return request.param
