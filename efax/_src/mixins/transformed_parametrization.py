from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import Any, Generic, cast, override

from array_api_compat import array_namespace
from jax import jacobian, vmap
from tjax import JaxArray, JaxComplexArray, JaxRealArray, Shape
from typing_extensions import TypeVar

from efax._src.expectation_parametrization import ExpectationParametrization
from efax._src.iteration import parameters
from efax._src.natural_parametrization import EP, NaturalParametrization

TEP = TypeVar("TEP", bound=ExpectationParametrization, default=Any)
NP = TypeVar("NP", bound=NaturalParametrization, default=Any)
Domain = TypeVar("Domain", bound=JaxComplexArray, default=JaxComplexArray)


class TransformedNaturalParametrization(
    NaturalParametrization[TEP, Domain],
    Generic[NP, EP, TEP, Domain],  # noqa: UP046
):
    """A NaturalParametrization defined by a differentiable transformation of a base distribution.

    Implements log_normalizer, to_exp, carrier_measure, and sufficient_statistics entirely in
    terms of the base distribution NP and the invertible sample transformation untransform_sample.
    Subclasses only need to implement the four abstract methods below.
    """

    @classmethod
    @abstractmethod
    def base_distribution_cls(cls) -> type[NP]:
        """Return the class of the base distribution."""
        raise NotImplementedError

    @abstractmethod
    def base_distribution(self) -> NP:
        """Return this distribution expressed in the base parametrization."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_expectation_from_base(cls, expectation_parametrization: EP) -> TEP:
        """Wrap the base expectation parameters as this distribution's expectation parameters."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def untransform_sample(cls, x: Domain, **fixed_parameters: JaxArray) -> Domain:
        """Map an observation from this distribution's domain back to the base domain.

        Must be differentiable with respect to x so that the Jacobian correction to the
        carrier measure can be computed automatically.
        """
        raise NotImplementedError

    @property
    @override
    def shape(self) -> Shape:
        return self.base_distribution().shape

    @override
    def log_normalizer(self) -> JaxRealArray:
        """The log-normalizer, delegated to the base distribution."""
        return self.base_distribution().log_normalizer()

    @override
    def to_exp(self) -> TEP:
        """The corresponding expectation parameters, delegated to the base distribution."""
        return self.create_expectation_from_base(self.base_distribution().to_exp())

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        """The carrier measure, adjusted by the log-absolute Jacobian of untransform_sample.

        Computed as base_distribution.carrier_measure(y) + log|det J_y(x)|, where y is
        the pre-image of x under the transformation.
        """
        xp = array_namespace(self, x)
        casted_x = cast("Domain", x)
        fixed_parameters = parameters(self, fixed=True, recurse=False)
        bound_fy = partial(self.untransform_sample, **fixed_parameters)
        y = bound_fy(casted_x)

        def log_abs_jac_y(x: JaxComplexArray) -> JaxRealArray:
            jac_y = jacobian(bound_fy)(x)
            if jac_y.ndim == 0:
                pass
            elif jac_y.ndim == 2:  # noqa: PLR2004
                jac_y = xp.linalg.det(jac_y)
            else:
                raise RuntimeError
            return xp.log(xp.abs(jac_y))

        for _ in range(self.ndim):
            log_abs_jac_y = vmap(log_abs_jac_y)

        # The carrier measure is the sum of:
        # * The base distribution's carrier measure applied to the base sample y, and
        # * log(abs(det(jac(y)))).
        return self.base_distribution().carrier_measure(y) + log_abs_jac_y(casted_x)

    @override
    @classmethod
    def sufficient_statistics(cls, x: Domain, **fixed_parameters: JaxArray) -> TEP:
        """Compute sufficient statistics by mapping x to the base domain first."""
        y = cls.untransform_sample(x, **fixed_parameters)
        base_cls = cls.base_distribution_cls()
        return cls.create_expectation_from_base(
            base_cls.sufficient_statistics(y, **fixed_parameters)
        )


TNP = TypeVar("TNP", bound=TransformedNaturalParametrization, default=Any)


class TransformedExpectationParametrization(ExpectationParametrization[TNP], Generic[EP, NP, TNP]):  # noqa: UP046
    """An ExpectationParametrization defined by a transformation of a base distribution.

    Implements to_nat entirely in terms of the base distribution EP and the natural
    parameter wrapper create_natural_from_base.  Subclasses only need to implement
    the three abstract methods below.
    """

    @classmethod
    @abstractmethod
    def base_distribution_cls(cls) -> type[EP]:
        """Return the class of the base distribution."""
        raise NotImplementedError

    @abstractmethod
    def base_distribution(self) -> EP:
        """Return this distribution expressed in the base parametrization."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_natural_from_base(cls, natural_parametrization: NP) -> TNP:
        """Wrap the base natural parameters as this distribution's natural parameters."""
        raise NotImplementedError

    @property
    @override
    def shape(self) -> Shape:
        return self.base_distribution().shape

    @override
    def to_nat(self) -> TNP:
        """The corresponding natural parameters, delegated to the base distribution."""
        return self.create_natural_from_base(self.base_distribution().to_nat())
