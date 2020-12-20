from __future__ import annotations

from functools import reduce
from typing import Any, Generic, Iterable, Tuple, Type, TypeVar, final, get_type_hints

from chex import Array
from jax import numpy as jnp
from tjax import RealArray, Shape, custom_jvp, field_values, jit

__all__ = ['NaturalParametrization', 'ExpectationParametrization']


EP = TypeVar('EP', bound='ExpectationParametrization[Any]')


class JitMethods:

    # Magic methods --------------------------------------------------------------------------------
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls.__name__ in ['VonMises', 'VonMisesFisher']:
            return

        # Apply jit.
        for name in ['log_normalizer',
                     'nat_to_exp',
                     'sufficient_statistics',
                     'cross_entropy',
                     'entropy',
                     'carrier_measure',
                     'expected_carrier_measure',
                     'pdf']:
            super_cls = super(cls, cls)
            if not hasattr(cls, name):
                continue
            original_method = getattr(cls, name)
            if hasattr(super_cls, name) and getattr(super_cls, name) is original_method:
                continue  # We only need to jit new methods.
            method = jit(original_method)
            setattr(cls, f'_original_{name}', method)

            if name != 'log_normalizer':
                setattr(cls, name, method)
                continue

            method_jvp: Any = custom_jvp(method)

            def ln_jvp(primals: Tuple[NaturalParametrization[Any]],
                       tangents: Tuple[NaturalParametrization[Any]]) -> Tuple[RealArray, RealArray]:
                q, = primals
                q_dot, = tangents
                y = q.log_normalizer()
                p = q.to_exp()
                y_dot = tree_dot_final(q_dot, p)
                return y, y_dot

            method_jvp.defjvp(ln_jvp)

            setattr(cls, name, method_jvp)


def dot_final(x: Array, y: Array, n_axes: int) -> RealArray:
    """
    Returns the real component of the dot product of the final n_axes axes of two arrays.
    """
    axes = tuple(range(-n_axes, 0))
    return jnp.real(jnp.sum(x * y, axis=axes))


def tree_dot_final(x: NaturalParametrization[Any], y: Any) -> RealArray:
    def dotted_fields() -> Iterable[Array]:
        for xf, yf, n_axes in zip(field_values(x, static=False),
                                  field_values(y, static=False),
                                  x.field_axes()):
            yield dot_final(xf, yf, n_axes)
    return reduce(jnp.add, dotted_fields())


class NaturalParametrization(JitMethods, Generic[EP]):
    """
    The natural parametrization of an exponential family distribution.
    """
    # Abstract methods -----------------------------------------------------------------------------
    def shape(self) -> Shape:
        raise NotImplementedError

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

    @classmethod
    def field_axes(cls) -> Iterable[int]:
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


NP = TypeVar('NP', bound=NaturalParametrization[Any])


class ExpectationParametrization(JitMethods, Generic[NP]):
    """
    The expectation parametrization of an exponential family distribution.  This class also doubles
    as the sufficient statistics of an exponential family distribution.
    """

    # Abstract methods -----------------------------------------------------------------------------
    def shape(self) -> Shape:
        raise NotImplementedError

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
