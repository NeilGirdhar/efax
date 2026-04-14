from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, final, get_type_hints

import jax
import jax.numpy as jnp
from array_api_compat import array_namespace
from jax import grad, jacfwd, vjp, vmap
from tjax import (
    JaxAbstractClass,
    JaxArray,
    JaxComplexArray,
    JaxRealArray,
    abstract_custom_jvp,
    abstract_jit,
    jit,
)
from tjax.dataclasses import dataclass
from typing_extensions import TypeVar

from .iteration import parameters
from .parametrization import Distribution
from .structure.assembler import Assembler
from .structure.estimator import Estimator
from .structure.flattener import Flattener
from .tools import parameter_dot_product

if TYPE_CHECKING:
    from .expectation_parametrization import ExpectationParametrization


EP = TypeVar("EP", bound="ExpectationParametrization", default=Any)
Domain = TypeVar("Domain", bound=JaxComplexArray | dict[str, Any], default=Any)


def _log_normalizer_jvp(
    primals: tuple[NaturalParametrization],
    tangents: tuple[NaturalParametrization],
) -> tuple[JaxRealArray, JaxRealArray]:
    """The log-normalizer's special JVP vastly improves numerical stability."""
    (q,) = primals
    (q_dot,) = tangents
    y = q.log_normalizer()
    p = q.to_exp()
    y_dot = parameter_dot_product(q_dot, p)
    return y, y_dot


@dataclass
class NaturalParametrization(Distribution, JaxAbstractClass, Generic[EP, Domain]):
    """The natural parametrization of an exponential family distribution.

    The motivation for the natural parametrization is combining and scaling independent predictive
    evidence.  In the natural parametrization, these operations correspond to scaling and addition.

    Class variables:
        characteristic_function_exact: True when `_complexify` shifts every natural parameter
            into the complex plane, so `characteristic_function` queries all sufficient
            statistics exactly.  Set to False in subclasses that must keep one or more
            parameters real (e.g. because `log_normalizer` calls a function such as
            ``gammaln`` that does not accept complex inputs); those components return 1
            instead of the true characteristic-function value.
    """

    characteristic_function_exact: ClassVar[bool] = True

    @abstract_custom_jvp(_log_normalizer_jvp)
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
        """Return the ExpectationParametrization class paired with this NaturalParametrization."""
        return get_type_hints(cls.to_exp)["return"]

    @jit
    @final
    def pdf(self, x: Domain) -> JaxRealArray:
        """Return the probability density (or mass) at x.

        Args:
            x: The sample.
        """
        xp = array_namespace(self)
        return xp.exp(self.log_pdf(x))

    @jit
    @final
    def log_pdf(self, x: Domain) -> JaxRealArray:
        """Return the log probability density (or log mass) at x.

        Args:
            x: The sample.
        """
        estimator = Estimator.from_natural(self)
        tx = estimator.sufficient_statistics(x)
        return parameter_dot_product(self, tx) - self.log_normalizer() + self.carrier_measure(x)

    @final
    def fisher_information_diagonal(self) -> Self:
        """The diagonal elements of the Fisher information.

        Returns: The Fisher information stored in a NaturalParametrization object whose fields
            are an array of the same shape as self.

        See also: apply_fisher_information.
        """
        xp = array_namespace(self)
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
        xp = array_namespace(self)
        fisher_information_diagonal = self.fisher_information_diagonal()
        structure = Assembler.create_assembler(self)
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
    def jeffreys_prior_density(self) -> JaxRealArray:
        """Return the Jeffreys prior density at this point: sqrt(det(Fisher information matrix))."""
        xp = array_namespace(self)
        fisher_matrix = self._fisher_information_matrix()
        return xp.sqrt(xp.linalg.det(fisher_matrix))

    @jit
    @final
    def apply_fisher_information(self, vector: EP) -> tuple[EP, Self]:
        """Apply the Fisher information matrix to a vector of expectation parameters.

        Computes both self.to_exp() and F(self) @ vector in a single VJP pass, which
        is more efficient than computing them separately.

        Args:
            vector: A set of expectation parameters to multiply by the Fisher information.

        Returns:
            The expectation parameters of self (self.to_exp()).
            The Fisher information of self applied to vector.
        """
        expectation_parameters: EP
        expectation_parameters, f_vjp = vjp(type(self).to_exp, self)
        return expectation_parameters, f_vjp(vector)

    @final
    def kl_divergence(self, q: Self) -> JaxRealArray:
        """Return the Kullback-Leibler divergence KL(self ‖ q)."""
        return self.to_exp().kl_divergence(q, self_nat=self)

    def _complexify(self, t: Self) -> Self:
        """Shift natural parameters into the complex plane: η → η + i·t.

        The default shifts every field, which is correct for any distribution
        whose ``log_normalizer`` accepts fully complex inputs.  Subclasses that
        must keep one or more parameters real (e.g. because ``log_normalizer``
        calls ``gammaln``, which rejects complex inputs) must override this
        method **and** set ``characteristic_function_exact = False`` on the
        class.  For those parameters, ``characteristic_function`` returns 1
        instead of the true E[exp(i·⟨t, T(x)⟩)] component.

        Args:
            t: Imaginary displacements, same pytree structure as self.

        Returns: A copy of self with each field replaced by ``η + i·t``.
        """
        return jax.tree_util.tree_map(lambda eta, delta: eta + 1j * delta, self, t)

    @jit
    @final
    def characteristic_function(self, t: Self) -> JaxComplexArray:
        """Characteristic function of the sufficient statistics: E[exp(i·⟨t, T(x)⟩)].

        For any exponential family, analytically continuing the log-normalizer
        gives the characteristic function of T(x):

            E[exp(i·⟨t, T(x)⟩)] = exp(A(η + i·t) − A(η))

        The frequency ``t`` lives in natural-parameter space and has the same
        pytree structure as ``self``: each field of ``t`` is the imaginary
        displacement for the corresponding natural parameter / sufficient
        statistic pair.

        Shapes::

            self : (*s,)           -- distribution with batch shape s
            t    : (*batch, *s)    -- frequency, broadcasts over self
            return: (*batch, *s)   -- one complex value per (batch, s) element

        Common usage: evaluate the CF at k frequencies simultaneously by
        giving ``t`` a leading batch dimension of size k::

            omegas = 2 * jnp.pi * jnp.fft.fftfreq(k, d=dx)
            t_grid = PoissonNP(omegas)           # shape (k,)
            phi = p.characteristic_function(t_grid)  # shape (k,)

        Note: when ``type(self).characteristic_function_exact`` is False, one
        or more natural parameters are kept real in ``_complexify`` because
        ``log_normalizer`` uses a function that rejects complex inputs.  The
        corresponding sufficient-statistic components return 1 rather than
        the true value.  Check the class attribute before relying on the
        result for those components.

        Args:
            t: Frequencies in natural-parameter space, same structure as self.

        Returns: Complex CF values, shape ``(*batch, *s)``.
        """
        shifted = self._complexify(t)
        return jnp.exp(shifted.log_normalizer() - self.log_normalizer())

    @classmethod
    def _flat_log_normalizer(
        cls,
        flattened_parameters: JaxRealArray,
        flattener: Flattener[Self],
    ) -> JaxRealArray:
        distribution = flattener.unflatten(flattened_parameters)
        return distribution.log_normalizer()

    @jit
    def _fisher_information_matrix(self) -> JaxRealArray:
        flattener, flattened = Flattener.flatten(self, mapped_to_plane=False)
        fisher_info_f = jacfwd(grad(self._flat_log_normalizer))
        for _ in range(len(self.shape)):
            fisher_info_f = vmap(fisher_info_f)
        return fisher_info_f(flattened, flattener)
