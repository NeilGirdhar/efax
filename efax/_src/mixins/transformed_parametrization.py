from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import Any, Generic, TypeVar, cast

from jax import jacobian, vmap
from tjax import JaxArray, JaxComplexArray, JaxRealArray, Shape
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..iteration import parameters
from ..natural_parametrization import EP, NaturalParametrization

TEP = TypeVar('TEP', bound=ExpectationParametrization[Any])
NP = TypeVar('NP', bound=NaturalParametrization[Any, Any])
Domain = TypeVar('Domain', bound=JaxComplexArray)


class TransformedNaturalParametrization(NaturalParametrization[TEP, Domain],
                                        Generic[NP, EP, TEP, Domain]):
    """Produce a NaturalParametrization by relating it to some base distrubtion NP."""
    @classmethod
    @abstractmethod
    def base_distribution_cls(cls) -> type[NP]:
        raise NotImplementedError

    @abstractmethod
    def base_distribution(self) -> NP:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_expectation_from_base(cls, expectation_parametrization: EP) -> TEP:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def sample_to_base_sample(cls, x: Domain, **fixed_parameters: JaxArray) -> Domain:
        raise NotImplementedError

    @property
    @override
    def shape(self) -> Shape:
        return self.base_distribution().shape

    @override
    def log_normalizer(self) -> JaxRealArray:
        """The log-normalizer."""
        return self.base_distribution().log_normalizer()

    @override
    def to_exp(self) -> TEP:
        """The corresponding expectation parameters."""
        return self.create_expectation_from_base(self.base_distribution().to_exp())

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        xp = self.array_namespace(x)
        casted_x = cast('Domain', x)
        fixed_parameters = parameters(self, fixed=True, recurse=False)
        bound_fy = partial(self.sample_to_base_sample, **fixed_parameters)
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
        # * The base distrubtion's carrier measure applied to the base sample y, and
        # * log(abs(det(jac(y)))).
        return self.base_distribution().carrier_measure(y) + log_abs_jac_y(casted_x)

    @override
    @classmethod
    def sufficient_statistics(cls, x: Domain, **fixed_parameters: JaxArray) -> TEP:
        y = cls.sample_to_base_sample(x, **fixed_parameters)
        base_cls = cls.base_distribution_cls()
        return cls.create_expectation_from_base(
                base_cls.sufficient_statistics(y, **fixed_parameters))


TNP = TypeVar('TNP', bound=TransformedNaturalParametrization[Any, Any, Any, Any])


class TransformedExpectationParametrization(ExpectationParametrization[TNP],
                                            Generic[EP, NP, TNP]):
    """Produce an ExpectationParametrization by relating it to some base distrubtion EP."""
    @classmethod
    @abstractmethod
    def base_distribution_cls(cls) -> type[EP]:
        raise NotImplementedError

    @abstractmethod
    def base_distribution(self) -> EP:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_natural_from_base(cls, natural_parametrization: NP) -> TNP:
        raise NotImplementedError

    @property
    @override
    def shape(self) -> Shape:
        return self.base_distribution().shape

    @override
    def to_nat(self) -> TNP:
        """The corresponding natural parameters."""
        return self.create_natural_from_base(self.base_distribution().to_nat())
