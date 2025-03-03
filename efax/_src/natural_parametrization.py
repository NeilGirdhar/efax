from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, final, get_type_hints

from jax import grad, jacfwd, vjp, vmap
from tjax import (JaxAbstractClass, JaxArray, JaxComplexArray, JaxRealArray, abstract_custom_jvp,
                  abstract_jit, jit)
from tjax.dataclasses import dataclass

from .iteration import parameters
from .parametrization import Distribution
from .structure import Flattener, MaximumLikelihoodEstimator, Structure
from .tools import parameter_dot_product

if TYPE_CHECKING:
    from .expectation_parametrization import ExpectationParametrization


EP = TypeVar('EP', bound='ExpectationParametrization[Any]')
Domain = TypeVar('Domain', bound=JaxComplexArray | dict[str, Any])


def log_normalizer_jvp(primals: tuple[NaturalParametrization[Any, Any]],
                       tangents: tuple[NaturalParametrization[Any, Any]],
                       ) -> tuple[JaxRealArray, JaxRealArray]:
    """The log-normalizer's special JVP vastly improves numerical stability."""
    q, = primals
    q_dot, = tangents
    y = q.log_normalizer()
    p = q.to_exp()
    y_dot = parameter_dot_product(q_dot, p)
    return y, y_dot


@dataclass
class NaturalParametrization(Distribution,
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
        Returns: The corresponding carrier measure, which is typically xp.zeros(x.shape[:-1]).
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def sufficient_statistics(cls, x: Domain, **fixed_parameters: JaxArray) -> EP:
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
        xp = self.array_namespace()
        return xp.exp(self.log_pdf(x))

    @jit
    @final
    def log_pdf(self, x: Domain) -> JaxRealArray:
        """The distribution's density or mass function at x.

        Args:
            x: The sample.
        """
        estimator = MaximumLikelihoodEstimator.create_estimator_from_natural(self)
        tx = estimator.sufficient_statistics(x)
        return parameter_dot_product(self, tx) - self.log_normalizer() + self.carrier_measure(x)

    @final
    def fisher_information_diagonal(self) -> Self:
        """The diagonal elements of the Fisher information.

        Returns: The Fisher information stored in a NaturalParametrization object whose fields
            are an array of the same shape as self.

        See also: apply_fisher_information.
        """
        xp = self.array_namespace()
        flattener, _ = Flattener.flatten(self)
        fisher_matrix = self._fisher_information_matrix()
        fisher_diagonal = xp.linalg.diagonal(fisher_matrix)
        return flattener.unflatten(fisher_diagonal)

    @final
    def fisher_information_trace(self) -> Self:
        """The trace of the Fisher information.

        Returns: The trace of the Fisher information stored in a NaturalParametrization object whose
            fields are scalar.

        See also: apply_fisher_information.
        """
        xp = self.array_namespace()
        fisher_information_diagonal = self.fisher_information_diagonal()
        structure = Structure.create(self)
        final_parameters = parameters(self)
        for path, (value, support) in parameters(fisher_information_diagonal, support=True).items():
            na = support.axes()
            if na == 0:
                new_value = value
            elif na == 1:
                new_value = xp.sum(value, axis=-1)
            elif na == 2:  # noqa: PLR2004
                new_value = xp.sum(xp.triu(value), axis=(-2, -1))
            else:
                raise RuntimeError
            final_parameters[path] = new_value
        return structure.assemble(final_parameters)

    @final
    def jeffreys_prior(self) -> JaxRealArray:
        xp = self.array_namespace()
        fisher_matrix = self._fisher_information_matrix()
        return xp.sqrt(xp.linalg.det(fisher_matrix))

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
                             flattener: Flattener[Self],
                             ) -> JaxRealArray:
        distribution = flattener.unflatten(flattened_parameters)
        return distribution.log_normalizer()

    @jit
    def _fisher_information_matrix(self) -> JaxRealArray:
        flattener, flattened = Flattener.flatten(self, map_to_plane=False)
        fisher_info_f = jacfwd(grad(self._flat_log_normalizer))
        for _ in range(len(self.shape)):
            fisher_info_f = vmap(fisher_info_f)
        return fisher_info_f(flattened, flattener)
