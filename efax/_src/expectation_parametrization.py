from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, final

import jax
import jax.numpy as jnp
from array_api_compat import array_namespace
from tjax import JaxComplexArray, JaxRealArray, jit
from typing_extensions import TypeVar

from .natural_parametrization import NaturalParametrization
from .parametrization import Distribution
from .structure.flattener import Flattener
from .tools import parameter_dot_product, parameter_map

NP = TypeVar("NP", bound=NaturalParametrization, default=Any)


class ExpectationParametrization(Distribution, Generic[NP]):
    """The expectation parametrization of an exponential family distribution.

    This class also doubles as the sufficient statistics of an exponential family distribution.

    The motivation for the expectation parametrization is combining independent and identically
    distributed observations into the maximum likelihood distribution.  In the expectation
    parametrization, this is an expected value.
    """

    @classmethod
    @abstractmethod
    def natural_parametrization_cls(cls) -> type[NP]:
        """Return the NaturalParametrization class paired with this ExpectationParametrization."""
        raise NotImplementedError

    @abstractmethod
    def to_nat(self) -> NP:
        """Return the corresponding natural parameters."""
        raise NotImplementedError

    @jit
    @final
    def kl_divergence(self, q: NP, *, self_nat: NP | None = None) -> JaxRealArray:
        """Return the Kullback-Leibler divergence KL(self ‖ q).

        Can be slow because it requires converting self to natural parameters.

        Args:
            q: The natural parameters of the second distribution.
            self_nat: The natural parameters of self, if already available, to avoid
                recomputing them.
        """
        if self_nat is None:
            self_nat = self.to_nat()
        xp = array_namespace(self)
        difference = parameter_map(xp.subtract, self_nat, q)
        return (
            parameter_dot_product(difference, self) + q.log_normalizer() - self_nat.log_normalizer()
        )


def expectation_parameters_from_characteristic_function(
    t: NaturalParametrization,
    cf_values: JaxComplexArray,
) -> ExpectationParametrization:
    """Estimate expectation parameters from characteristic function values via OLS.

    The imaginary part of the log-CF is linear in the frequencies to first order:

        Im(log φ_Y(t)) ≈ ⟨t, E[T(x)]⟩

    where each field of ``t`` is a frequency for the corresponding sufficient
    statistic, and ``E[T(x)]`` are the expectation parameters.  Stacking ``k``
    frequency evaluations gives an overdetermined linear system solved by OLS.

    This estimate is **exact for Normal** (log φ is exactly linear in t) and a
    first-order approximation for other families — accuracy improves as the
    magnitude of the frequencies in ``t`` decreases.

    Works for any NP field shape (scalar, vector, or matrix).  Each leaf of ``t``
    has shape ``(*s, k, *field_shape)`` where ``*s`` is the distribution batch
    shape and ``*field_shape`` is the per-element shape of that NP field.  All
    field dimensions are flattened into a single column dimension for the OLS.

    Shapes::

        t         : (*s, k)  -- NP with t.shape=(*s,k); leaves (*s,k,*field_shape)
        cf_values : (*s, k)  -- complex CF values at those frequencies
        return    : (*s,)    -- estimated EP, one per batch element

    Args:
        t: Frequency grid.  ``t.shape`` must equal ``(*s, k)``.  Each leaf has
           shape ``(*s, k, *field_shape)`` where the ``k`` columns span the NP
           field directions for well-conditioned recovery.
        cf_values: Complex CF values ``φ_Y(tⱼ)`` at the frequencies in ``t``,
            shape ``(*s, k)``.

    Returns: ExpectationParametrization with shape ``(*s,)`` whose fields are
        the OLS estimates of ``E[T(x)]``.
    """
    _, design = Flattener.flatten(t, mapped_to_plane=False)

    # Response: Im(log φ), shape (*s, k)
    b = jnp.imag(jnp.log(cf_values))

    # OLS via normal equations: (designᵀ design) chi = designᵀ b
    gram = jnp.einsum("...ki,...kj->...ij", design, design)  # (*s, total_dim, total_dim)
    rhs = jnp.einsum("...ki,...k->...i", design, b)  # (*s, total_dim)
    chi = jnp.linalg.solve(gram, rhs[..., jnp.newaxis]).squeeze(-1)  # (*s, total_dim)

    # Reconstruct EP: use sufficient_statistics on a dummy to get an EP flattener.
    leaves = jax.tree_util.tree_leaves(t)
    n_t = len(t.shape)
    first_field_shape = leaves[0].shape[n_t:]
    dummy_ep = type(t).sufficient_statistics(jnp.zeros(first_field_shape))
    ep_flattener = Flattener.create_flattener(dummy_ep, mapped_to_plane=False)
    return ep_flattener.unflatten(chi)
