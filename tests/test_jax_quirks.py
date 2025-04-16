import jax._src.xla_bridge as xb  # noqa: PLC2701

import efax  # noqa: F401


def jax_is_initialized() -> bool:
    return bool(xb._backends)  # noqa: SLF001  # pyright: ignore


def test_jax_not_initialized() -> None:
    assert not jax_is_initialized()
