import jax._src.xla_bridge as xb  # ruff:ignore[import-private-name]
import pytest

import efax  # ruff:ignore[unused-import]


def jax_is_initialized() -> bool:
    return bool(xb._backends)  # ruff:ignore[private-member-access]


@pytest.mark.run(order=1)
@pytest.mark.nondistribution
def test_jax_not_initialized() -> None:
    assert not jax_is_initialized()
