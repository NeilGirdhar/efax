from __future__ import annotations

from typing import Any, Generic, TypeVar, cast

import jax.numpy as jnp
from tjax import JaxComplexArray, JaxRealArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces import Multidimensional
from ..natural_parametrization import NaturalParametrization
from ..parameter import VectorSupport
from .has_entropy import HasEntropyEP

__all__: list[str] = []


EP = TypeVar('EP', bound='DiagonalizedEP[Any]')
Domain = TypeVar('Domain', bound=JaxComplexArray)


@dataclass
class DiagonalizedNP(Multidimensional,
                     NaturalParametrization[EP, Domain],
                     Generic[EP, Domain]):
    """Produce a multi-dimensional version of a natural parametrization."""
    base_distribution: NaturalParametrization[Any, Any]

    @property
    @override
    def shape(self) -> Shape:
        return self.base_distribution.shape[:-1]

    @override
    def domain_support(self) -> VectorSupport:
        return VectorSupport()

    @override
    def log_normalizer(self) -> JaxRealArray:
        return jnp.sum(self.base_distribution.log_normalizer(), axis=-1)

    @override
    def to_exp(self) -> EP:
        return cast(EP, DiagonalizedEP(self.base_distribution.to_exp()))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.sum(self.base_distribution.carrier_measure(x), axis=-1)

    @override
    def sufficient_statistics(self, x: Domain) -> EP:
        return cast(EP, DiagonalizedEP(self.base_distribution.sufficient_statistics(x)))

    @override
    def dimensions(self) -> int:
        return self.base_distribution.shape[-1]


NP = TypeVar('NP', bound=DiagonalizedNP[Any, Any])


@dataclass
class DiagonalizedEP(Multidimensional, ExpectationParametrization[NP], Generic[NP]):
    """Produce a multi-dimensional version of an expectation parametrization."""
    base_distribution: ExpectationParametrization[Any]

    @property
    @override
    def shape(self) -> Shape:
        return self.base_distribution.shape[:-1]

    @override
    def domain_support(self) -> VectorSupport:
        return VectorSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[NP]:
        return cast(type[NP], DiagonalizedNP)

    @override
    def to_nat(self) -> NP:
        return cast(NP, DiagonalizedNP(self.base_distribution.to_nat()))

    @override
    def dimensions(self) -> int:
        return self.base_distribution.shape[-1]


class HasEntropyDEP(HasEntropyEP[NP], DiagonalizedEP[NP], Generic[NP]):
    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        assert isinstance(self.base_distribution, HasEntropyEP)
        return jnp.sum(self.base_distribution.expected_carrier_measure(), axis=-1)


class SamplableDEP(Samplable, DiagonalizedEP[NP], Generic[NP]):
    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        assert isinstance(self.base_distribution, Samplable)
        return self.base_distribution.sample(key, shape)


class HasGCP_DEP(Samplable, DiagonalizedEP[NP], Generic[NP]):
    @override
    def generalized_conjugate_prior_distribution(self, n: JaxRealArray
                                                 ) -> NaturalParametrization[Any]:
        assert isinstance(self.base_distribution, HasConjugatePrior)
        return DiagonalizedNP(self.base_distribution.conjugate_prior_distribution(n))

    @override
    def conjugate_prior_observation(self) -> JaxRealArray:
        assert isinstance(self.base_distribution, HasConjugatePrior)
        return self.base_distribution.conjugate_prior_observation()
