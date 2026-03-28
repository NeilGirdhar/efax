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
        raise NotImplementedError

    @abstractmethod
    def to_nat(self) -> NP:
        """The corresponding natural parameters."""
        raise NotImplementedError

    @jit
    @final
    def kl_divergence(self, q: NP, *, self_nat: NP | None = None) -> JaxRealArray:
        """The Kullback-Leibler divergence.

        This can be quite slow since it depends on a conversion to natural parameters.

        Args:
            self: The expectation parameters of the first distribution.
            q: The natural parameters of second destribution.
            self_nat: The natural parameters of the first distribution, if available.
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

    Shapes::

        t         : (*s, k)  -- NP whose fields are the k frequency vectors
        cf_values : (*s, k)  -- complex CF values at those frequencies
        return    : (*s,)    -- estimated EP, one per batch element

    Args:
        t: Frequency grid.  Each field has shape ``(*s, k)``.  The k columns
           are the frequency vectors at which ``characteristic_function`` was
           evaluated; different columns should span the NP field directions for
           well-conditioned recovery.
        cf_values: Complex CF values ``φ_Y(tⱼ)`` at the frequencies in ``t``,
            shape ``(*s, k)``.

    Returns: ExpectationParametrization with shape ``(*s,)`` whose fields are
        the OLS estimates of ``E[T(x)]``.
    """
    leaves = jax.tree_util.tree_leaves(t)
    n = len(leaves)

    # Design matrix: each column is one NP field evaluated at k frequencies
    # shape (*s, k, n)
    design = jnp.stack(leaves, axis=-1)

    # Response: Im(log φ), shape (*s, k)
    b = jnp.imag(jnp.log(cf_values))

    # OLS via normal equations: (designᵀ design) chi = designᵀ b
    gram = jnp.einsum("...ki,...kj->...ij", design, design)  # (*s, n, n)
    rhs = jnp.einsum("...ki,...k->...i", design, b)  # (*s, n)
    chi = jnp.linalg.solve(gram, rhs)  # (*s, n)

    # Reconstruct EP: use sufficient_statistics on a dummy to get the treedef,
    # then inject the OLS solution into the leaves.
    dummy_ep = type(t).sufficient_statistics(jnp.zeros(1))
    _, ep_treedef = jax.tree_util.tree_flatten(dummy_ep)
    return ep_treedef.unflatten([chi[..., i] for i in range(n)])
