import pytest

from .create_info import create_infos


@pytest.fixture(scope='session',
                params=create_infos())
def distribution_info(request):
    return request.param
