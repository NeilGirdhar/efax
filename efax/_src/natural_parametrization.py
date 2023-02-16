from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Generic, TypeVar, final, get_type_hints

import jax.numpy as jnp
from jax import grad, jacfwd, vjp, vmap
from tjax import ComplexArray, RealArray, jit
from typing_extensions import Self

from .parametrization import Parametrization
from .tools import parameters_dot_product

if TYPE_CHECKING:
    from .expectation_parametrization import ExpectationParametrization

__all__ = ['NaturalParametrization']


EP = TypeVar('EP', bound='ExpectationParametrization[Any]')
Domain = TypeVar('Domain', bound=ComplexArray)


class NaturalParametrization(Parametrization, Generic[EP, Domain]):
    """The natural parametrization of an exponential family distribution.

    The motivation for the natural parametrization is combining and scaling independent predictive
    evidence.  In the natural parametrization, these operations correspond to scaling and addition.
    """
    # Abstract methods -----------------------------------------------------------------------------
    def log_normalizer(self) -> RealArray:
        """Returns: The log-normalizer."""
        raise NotImplementedError

    def to_exp(self) -> EP:
        """Returns: The corresponding expectation parameters."""
        raise NotImplementedError

    def carrier_measure(self, x: Domain) -> RealArray:
        """The corresponding carrier measure.

        Args:
            x: The sample.
        Returns: The corresponding carrier measure, which is typically jnp.zeros(x.shape[:-1]).
        """
        raise NotImplementedError

    def sufficient_statistics(self, x: Domain) -> EP:
        """The corresponding sufficient statistics.

        Args:
            x: The sample.
        Returns: These are conveniently stored as expectation parameters.
        """
        raise NotImplementedError

    # Final methods --------------------------------------------------------------------------------
    @classmethod
    def expectation_parametrization_cls(cls) -> type[EP]:
        return get_type_hints(cls.to_exp)['return']

    @jit
    @final
    def entropy(self) -> RealArray:
        """The Shannon entropy."""
        return self.to_exp().cross_entropy(self)

    @jit
    @final
    def pdf(self, x: Domain) -> RealArray:
        """The distribution's density or mass function at x.

        Args:
            x: The sample.
        """
        tx = self.sufficient_statistics(x)
        return jnp.exp(parameters_dot_product(self, tx) - self.log_normalizer()
                       + self.carrier_measure(x))

    @final
    def fisher_information_diagonal(self: NaturalParametrization[EP, Domain]) -> (
            NaturalParametrization[EP, Domain]):
        """The diagonal elements of the Fisher information.

        Returns: The Fisher information stored in a NaturalParametrization object whose fields
            are an array of the same shape as self.

        See also: apply_fisher_information.
        """
        fisher_matrix = self._fisher_information_matrix()
        fisher_diagonal = jnp.diagonal(fisher_matrix)
        return type(self).unflattened(fisher_diagonal, **self.fixed_parameters_mapping())

    @final
    def fisher_information_trace(self: NaturalParametrization[EP, Domain]) -> (
            NaturalParametrization[EP, Domain]):
        """The trace of the Fisher information.

        Returns: The trace of the Fisher information stored in a NaturalParametrization object whose
            fields are scalar.

        See also: apply_fisher_information.
        """
        fisher_information_diagonal = self.fisher_information_diagonal()
        kwargs = {}
        for name, value, support in fisher_information_diagonal.parameters_name_value_support():
            na = support.axes()
            if na == 0:
                new_value = value
            elif na == 1:
                new_value = jnp.sum(value, axis=-1)
            elif na == 2:  # noqa: PLR2004
                new_value = jnp.sum(jnp.triu(value), axis=(-2, -1))
            else:
                raise RuntimeError
            kwargs[name] = new_value
        return replace(self, **kwargs)

    @final
    def jeffreys_prior(self) -> RealArray:
        fisher_matrix = self._fisher_information_matrix()
        return jnp.sqrt(jnp.linalg.det(fisher_matrix))

    @jit
    @final
    def apply_fisher_information(self: NaturalParametrization[EP, Domain],
                                 vector: EP) -> tuple[EP, NaturalParametrization[EP, Domain]]:
        """Efficiently apply the Fisher information matrix to a vector.

        Args:
            vector: Some set of expectation parameters.

        Returns:
            The expectation parameters corresponding to self.
            The Fisher information of self applied to the inputted vector.
        """
        expectation_parameters: EP
        expectation_parameters, f_vjp = vjp(type(self).to_exp, self)
        return expectation_parameters, f_vjp(vector)

    @final
    def kl_divergence(self, q: Self) -> RealArray:
        return self.to_exp().kl_divergence(q, self_nat=self)

    # Private methods ------------------------------------------------------------------------------
    @classmethod
    def _flat_log_normalizer(cls, flattened_parameters: RealArray, **kwargs: Any) -> RealArray:
        distribution = cls.unflattened(flattened_parameters, **kwargs)
        return distribution.log_normalizer()

    @jit
    def _fisher_information_matrix(self: NaturalParametrization[EP, Domain]) -> RealArray:
        fisher_info_f = jacfwd(grad(self._flat_log_normalizer))
        for _ in range(len(self.shape)):
            fisher_info_f = vmap(fisher_info_f)
        return fisher_info_f(self.flattened(), **self.fixed_parameters_mapping())
