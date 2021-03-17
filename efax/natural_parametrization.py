from __future__ import annotations

from functools import partial
from typing import (TYPE_CHECKING, Any, Callable, Generic, Tuple, Type, TypeVar, final,
                    get_type_hints)

import jax.numpy as jnp
from chex import Array
from jax import grad, jacfwd, vjp, vmap
from tjax import RealArray, jit

from .parameter import parameter_names_values_support
from .parametrization import Parametrization
from .tools import tree_dot_final

__all__ = ['NaturalParametrization']

EP = TypeVar('EP', bound='ExpectationParametrization[Any]')


class NaturalParametrization(Parametrization, Generic[EP]):
    """
    The natural parametrization of an exponential family distribution.

    The motivation for the natural parametrization is combining and scaling independent predictive
    evidence.  In the natural parametrization, these operations correspond to scaling and addition.
    """
    T = TypeVar('T', bound='NaturalParametrization[EP]')

    # Abstract methods -----------------------------------------------------------------------------
    def log_normalizer(self) -> RealArray:
        """
        Returns: The log-normalizer.
        """
        raise NotImplementedError

    def to_exp(self) -> EP:
        """
        Returns: The corresponding expectation parameters.
        """
        raise NotImplementedError

    def carrier_measure(self, x: Array) -> RealArray:
        """
        Args:
            x: The sample.
        Returns: The corresponding carrier measure, which is typically jnp.zeros(x.shape[:-1]).
        """
        raise NotImplementedError

    # Abstract class methods -----------------------------------------------------------------------
    def sufficient_statistics(self, x: Array) -> EP:
        """
        Args:
            x: The sample.
        Returns: The corresponding sufficient statistics, which are conveniently stored as
            expectation parameters.
        """
        raise NotImplementedError

    # Final methods --------------------------------------------------------------------------------
    @classmethod
    def expectation_parametrization_cls(cls) -> Type[EP]:
        return get_type_hints(cls.to_exp)['return']

    @jit
    @final
    def entropy(self) -> RealArray:
        """
        Returns: The Shannon entropy.
        """
        return self.to_exp().cross_entropy(self)

    @jit
    @final
    def pdf(self, x: Array) -> RealArray:
        """
        Args:
            x: The sample.
        Returns: The distribution's density or mass function at x.
        """
        tx = self.sufficient_statistics(x)
        return jnp.exp(tree_dot_final(self, tx) - self.log_normalizer() + self.carrier_measure(x))

    @final
    def fisher_information(self: T, diagonal: bool = False, trace: bool = False) -> T:
        """
        Args:
            diagonal: If true, return only the diagonal elements of the Fisher information matrices.
            trace: If true, return the trace of the Fisher information matrices.
        Returns: The Fisher information stored in a NaturalParametrization object R whose fields
            are:
            * A scalar if trace is true.
            * An array of the same shape as self if diagonal is true.
            * Otherwise, a NaturalParametrization object whose fields are arrays.
        See also: apply_fisher_information
        """
        fisher_information = self._calculate_fisher_information(len(self.shape()))

        if not trace and not diagonal:
            return fisher_information

        kwargs = {}
        f = jnp.trace if trace else jnp.diagonal
        for name, value, support in parameter_names_values_support(fisher_information):
            kwargs[name] = _summarize_fisher_information(f, getattr(value, name), support.axes())
        return fisher_information.replace(**kwargs)

    @jit
    @final
    def apply_fisher_information(self: T, vector: EP) -> Tuple[EP, T]:
        """
        Args:
            vector: Some set of expectation parameters.
        Returns:
            The expectation parameters corresponding to self.
            The Fisher information of self applied to the inputted vector.
        """
        expectation_parameters: EP
        expectation_parameters, f_vjp = vjp(type(self).to_exp, self)
        return expectation_parameters, f_vjp(vector)

    # Private methods ------------------------------------------------------------------------------
    @partial(jit, static_argnums=1)
    def _calculate_fisher_information(self: T, len_shape: int) -> T:
        fisher_info_f = jacfwd(grad(type(self).log_normalizer))
        for _ in range(len_shape):
            fisher_info_f = vmap(fisher_info_f)
        return fisher_info_f(self)


def _summarize_fisher_information(f: Callable[..., Array], array: Array, axes: int) -> Array:
    if axes == 0:
        return array
    if axes == 1:
        return f(array, axis1=-2, axis2=-1)
    if axes == 2:
        return f(f(array, axis1=-3, axis2=-1), axis1=-2, axis2=-1)
    raise ValueError


if TYPE_CHECKING:
    from .expectation_parametrization import ExpectationParametrization

    # Prevent autoflake from removing imports.
    # pylint: disable=pointless-statement
    ExpectationParametrization, Any
