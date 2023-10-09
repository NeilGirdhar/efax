from __future__ import annotations

from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
from jax.nn import softplus
from jax.scipy import special as jss
from tjax import JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.multidimensional import Multidimensional
from ..interfaces.samplable import Samplable
from ..mixins.exp_to_nat import ExpToNat
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import VectorSupport, distribution_parameter

__all__: list[str] = []


EP = TypeVar('EP', bound='DirichletCommonEP[Any]')


@dataclass
class DirichletCommonNP(HasEntropyNP[EP],
                        Samplable,
                        Multidimensional,
                        NaturalParametrization[EP, JaxRealArray],
                        Generic[EP]):
    alpha_minus_one: JaxRealArray = distribution_parameter(VectorSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.alpha_minus_one.shape[:-1]

    @override
    def log_normalizer(self) -> JaxRealArray:
        q = self.alpha_minus_one
        return (jnp.sum(jss.gammaln(q + 1.0), axis=-1)
                - jss.gammaln(jnp.sum(q, axis=-1) + self.dimensions()))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        return jax.random.dirichlet(key, 1.0 + self.alpha_minus_one, shape)[..., :-1]

    @override
    def dimensions(self) -> int:
        return self.alpha_minus_one.shape[-1]

    def _exp_helper(self) -> JaxRealArray:
        q = self.alpha_minus_one
        return jss.digamma(q + 1.0) - jss.digamma(jnp.sum(q, axis=-1, keepdims=True) + q.shape[-1])


NP = TypeVar('NP', bound=DirichletCommonNP[Any])


@dataclass
class DirichletCommonEP(HasEntropyEP[NP],
                        ExpToNat[NP, JaxRealArray],
                        Multidimensional, Generic[NP]):
    mean_log_probability: JaxRealArray = distribution_parameter(VectorSupport())

    @property
    @override
    def shape(self) -> Shape:
        return self.mean_log_probability.shape[:-1]

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        return jnp.zeros(self.shape)

    @override
    def initial_search_parameters(self) -> JaxRealArray:
        return jnp.zeros(self.mean_log_probability.shape)

    @override
    def search_gradient(self, search_parameters: JaxRealArray) -> JaxRealArray:
        return self._natural_gradient(self.search_to_natural(search_parameters)).alpha_minus_one

    @override
    def dimensions(self) -> int:
        return self.mean_log_probability.shape[-1]

    @classmethod
    def _transform_nat_helper(cls, search_parameters: JaxRealArray) -> JaxRealArray:
        # Run Newton's method on the whole real hyperspace.
        return softplus(search_parameters) - 1.0
