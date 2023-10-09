from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Generic, TypeVar, final, get_type_hints

import jax.numpy as jnp
from jax import grad, jacfwd, vjp, vmap
from tjax import (JaxAbstractClass, JaxComplexArray, JaxRealArray, abstract_custom_jvp,
                  abstract_jit, jit)
from tjax.dataclasses import DataclassInstance, dataclass
from typing_extensions import Self

from .parametrization import Parametrization
from .tools import parameters_dot_product

if TYPE_CHECKING:
    from .expectation_parametrization import ExpectationParametrization

__all__ = ['NaturalParametrization']


EP = TypeVar('EP', bound='ExpectationParametrization[Any]')
Domain = TypeVar('Domain', bound=JaxComplexArray)


def log_normalizer_jvp(primals: tuple[NaturalParametrization[Any, Any]],
                       tangents: tuple[NaturalParametrization[Any, Any]],
                       ) -> tuple[JaxRealArray, JaxRealArray]:
    """The log-normalizer's special JVP vastly improves numerical stability."""
    q, = primals
    q_dot, = tangents
    y = q.log_normalizer()
    p = q.to_exp()
    y_dot = parameters_dot_product(q_dot, p)
    return y, y_dot


@dataclass
class NaturalParametrization(Parametrization,
                             JaxAbstractClass,
                             Generic[EP, Domain]):
    """The natural parametrization of an exponential family distribution.

    The motivation for the natural parametrization is combining and scaling independent predictive
    evidence.  In the natural parametrization, these operations correspond to scaling and addition.
    """
    @abstract_custom_jvp(log_normalizer_jvp)
    @abstract_jit
    @abstractmethod
    def log_normalizer(self) -> JaxRealArray:
        """Returns: The log-normalizer."""
        raise NotImplementedError

    @abstract_jit
    @abstractmethod
    def to_exp(self) -> EP:
        """Returns: The corresponding expectation parameters."""
        raise NotImplementedError

    @abstract_jit
    @abstractmethod
    def carrier_measure(self, x: Domain) -> JaxRealArray:
        """The corresponding carrier measure.

        Args:
            x: The sample.
        Returns: The corresponding carrier measure, which is typically jnp.zeros(x.shape[:-1]).
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def sufficient_statistics(cls, x: Domain, **fixed_parameters: Any) -> EP:
        """The sufficient statistics corresponding to an observation.

        This is typically used in maximum likelihood estimation.

        Args:
            x: The samples.
            fixed_parameters: The fixed parameters of the expectation parametrization.
        Returns: The sufficient statistics, stored as expectation parameters.
        """
        raise NotImplementedError

    @classmethod
    def expectation_parametrization_cls(cls) -> type[EP]:
        return get_type_hints(cls.to_exp)['return']

    @jit
    @final
    def pdf(self, x: Domain) -> JaxRealArray:
        """The distribution's density or mass function at x.

        Args:
            x: The sample.
        """
        return jnp.exp(self.log_pdf(x))

    @jit
    @final
    def log_pdf(self, x: Domain) -> JaxRealArray:
        """The distribution's density or mass function at x.

        Args:
            x: The sample.
        """
        tx = self.sufficient_statistics(x, **self.fixed_parameters())
        return parameters_dot_product(self, tx) - self.log_normalizer() + self.carrier_measure(x)

    @final
    def fisher_information_diagonal(self) -> Self:
        """The diagonal elements of the Fisher information.

        Returns: The Fisher information stored in a NaturalParametrization object whose fields
            are an array of the same shape as self.

        See also: apply_fisher_information.
        """
        fisher_matrix = self._fisher_information_matrix()
        fisher_diagonal = jnp.diagonal(fisher_matrix)
        return type(self).unflattened(fisher_diagonal, **self.fixed_parameters())

    @final
    def fisher_information_trace(self) -> Self:
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
        assert isinstance(self, DataclassInstance)
        return replace(self, **kwargs)  # type: ignore[misc, return-value]

    @final
    def jeffreys_prior(self) -> JaxRealArray:
        fisher_matrix = self._fisher_information_matrix()
        return jnp.sqrt(jnp.linalg.det(fisher_matrix))

    @jit
    @final
    def apply_fisher_information(self, vector: EP) -> tuple[EP, Self]:
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
    def kl_divergence(self, q: Self) -> JaxRealArray:
        return self.to_exp().kl_divergence(q, self_nat=self)

    @classmethod
    def _flat_log_normalizer(cls,
                             flattened_parameters: JaxRealArray,
                             **fixed_parameters: Any,
                             ) -> JaxRealArray:
        distribution = cls.unflattened(flattened_parameters, **fixed_parameters)
        return distribution.log_normalizer()

    @jit
    def _fisher_information_matrix(self) -> JaxRealArray:
        fisher_info_f = jacfwd(grad(self._flat_log_normalizer))
        for _ in range(len(self.shape)):
            fisher_info_f = vmap(fisher_info_f)
        return fisher_info_f(self.flattened(), **self.fixed_parameters())
