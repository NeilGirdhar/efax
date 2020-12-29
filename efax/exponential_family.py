from __future__ import annotations

from functools import partial, reduce
from typing import Any, Callable, Generic, Iterable, Tuple, Type, TypeVar, final, get_type_hints

from chex import Array
from jax import grad, jacfwd
from jax import numpy as jnp
from jax import vjp, vmap
from tjax import RealArray, field_values, jit

from .parameter import parameter_names_values_axes
from .parametrization import Parametrization

__all__ = ['NaturalParametrization', 'ExpectationParametrization']


EP = TypeVar('EP', bound='ExpectationParametrization[Any]')


def dot_final(x: Array, y: Array, n_axes: int) -> RealArray:
    """
    Returns the real component of the dot product of the final n_axes axes of two arrays.
    """
    axes = tuple(range(-n_axes, 0))
    return jnp.real(jnp.sum(x * y, axis=axes))


def tree_dot_final(x: NaturalParametrization[Any], y: Any) -> RealArray:
    def dotted_fields() -> Iterable[Array]:
        for (_, xf, n_axes), yf in zip(parameter_names_values_axes(x),
                                       field_values(y, static=False)):
            yield dot_final(xf, yf, n_axes)
    return reduce(jnp.add, dotted_fields())


class NaturalParametrization(Parametrization, Generic[EP]):
    """
    The natural parametrization of an exponential family distribution.
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
        Returns: The corresponding sufficient statistics.
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
        Returns: The entropy.
        """
        return self.to_exp().cross_entropy(self)

    @jit
    @final
    def pdf(self, x: Array) -> RealArray:
        """
        Args:
            x: The sample.
        Returns: The distribution's density or mass function at x.shape.
        """
        tx = self.sufficient_statistics(x)
        return jnp.exp(tree_dot_final(self, tx)
                       - self.log_normalizer()
                       + self.carrier_measure(x))

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
        fisher_information = self._fisher_helper(len(self.shape()))

        if not trace and not diagonal:
            return fisher_information

        kwargs = {}
        f = jnp.trace if trace else jnp.diagonal
        for name, value, axes in parameter_names_values_axes(fisher_information):
            kwargs[name] = transform(f, getattr(value, name), axes)
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
    def _fisher_helper(self: T, len_shape: int) -> T:
        fisher_info_f = jacfwd(grad(type(self).log_normalizer))
        for _ in range(len_shape):
            fisher_info_f = vmap(fisher_info_f)
        return fisher_info_f(self)


NP = TypeVar('NP', bound=NaturalParametrization[Any])


def transform(f: Callable[..., Array], array: Array, axes: int) -> Array:
    if axes == 0:
        return array
    if axes == 1:
        return f(array, axis1=-2, axis2=-1)
    if axes == 2:
        return f(f(array, axis1=-3, axis2=-1), axis1=-2, axis2=-1)
    raise ValueError


class ExpectationParametrization(Parametrization, Generic[NP]):
    """
    The expectation parametrization of an exponential family distribution.  This class also doubles
    as the sufficient statistics of an exponential family distribution.
    """
    # Abstract methods -----------------------------------------------------------------------------
    def to_nat(self) -> NP:
        """
        Returns: The corresponding natural parameters.
        """
        raise NotImplementedError

    def expected_carrier_measure(self) -> RealArray:
        """
        Returns: The expected carrier measure of the distribution.  This is the missing term from
            the inner product between the observed distribution and the predicted distribution.
            Often, it is just jnp.zeros(self.shape).
        """
        raise NotImplementedError

    # Final methods --------------------------------------------------------------------------------
    @classmethod
    def natural_parametrization_cls(cls) -> Type[NP]:
        return get_type_hints(cls.to_nat)['return']

    @jit
    @final
    def cross_entropy(self, q: NP) -> RealArray:
        """
        Args:
            q: The natural parameters of the prediction.
        Returns:
            The cross entropy.
        """
        return (-tree_dot_final(q, self)
                + q.log_normalizer()
                - self.expected_carrier_measure())
