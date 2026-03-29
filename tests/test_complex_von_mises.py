from __future__ import annotations

import jax.numpy as jnp
from tjax import assert_tree_allclose

from efax import ComplexVonMisesNP, VonMisesFisherNP


def test_complex_von_mises_matches_von_mises_fisher() -> None:
    q_vmf = VonMisesFisherNP(jnp.asarray([[1.2, -0.7], [0.0, 0.0]]))
    q_cvm = ComplexVonMisesNP(
        q_vmf.mean_times_concentration[..., 0] + 1j * q_vmf.mean_times_concentration[..., 1]
    )

    p_vmf = q_vmf.to_exp()
    p_cvm = q_cvm.to_exp()

    assert_tree_allclose(q_cvm.kappa(), q_vmf.kappa())
    assert_tree_allclose(q_cvm.angle(), q_vmf.to_kappa_angle()[1])
    assert_tree_allclose(
        jnp.real(p_cvm.mean),
        p_vmf.mean[..., 0],
    )
    assert_tree_allclose(
        jnp.imag(p_cvm.mean),
        p_vmf.mean[..., 1],
    )
