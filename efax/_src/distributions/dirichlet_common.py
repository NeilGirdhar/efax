from __future__ import annotations

from typing import Any, Generic, List, Optional, TypeVar

import jax
import jax.numpy as jnp
from jax.nn import softplus
from jax.scipy import special as jss
from tjax import Generator, RealArray, Shape
from tjax.dataclasses import dataclass

from ..exp_to_nat import ExpToNat
from ..natural_parametrization import EP, NaturalParametrization
from ..parameter import VectorSupport, distribution_parameter
from ..samplable import Samplable

__all__: List[str] = []


# https://github.com/python/mypy/issues/10140
# EP = TypeVar('EP', bound='DirichletCommonEP[Any]')


@dataclass
class DirichletCommonNP(NaturalParametrization[EP, RealArray], Samplable, Generic[EP]):
    alpha_minus_one: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.alpha_minus_one.shape[:-1]

    def log_normalizer(self) -> RealArray:
        q = self.alpha_minus_one
        return (jnp.sum(jss.gammaln(q + 1.0), axis=-1)
                - jss.gammaln(jnp.sum(q, axis=-1) + q.shape[-1]))

    def sample(self, rng: Generator, shape: Optional[Shape] = None) -> RealArray:
        if shape is not None:
            shape += self.shape
        return jax.random.dirichlet(rng.key, 1.0 + self.alpha_minus_one, shape)[..., :-1]

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.alpha_minus_one.shape[-1]

    # Private methods ------------------------------------------------------------------------------
    def _exp_helper(self) -> RealArray:
        q = self.alpha_minus_one
        return jss.digamma(q + 1.0) - jss.digamma(jnp.sum(q, axis=-1, keepdims=True) + q.shape[-1])


NP = TypeVar('NP', bound=DirichletCommonNP[Any])


@dataclass
class DirichletCommonEP(ExpToNat[NP, RealArray], Generic[NP]):
    mean_log_probability: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean_log_probability.shape[:-1]

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape)

    def initial_search_parameters(self) -> RealArray:
        return jnp.zeros(self.mean_log_probability.shape)

    def search_gradient(self, search_parameters: RealArray) -> RealArray:
        return self._natural_gradient(self.search_to_natural(search_parameters)).alpha_minus_one

    # New methods ----------------------------------------------------------------------------------
    def dimensions(self) -> int:
        return self.mean_log_probability.shape[-1]

    # Private methods ------------------------------------------------------------------------------
    @classmethod
    def _transform_nat_helper(cls, search_parameters: RealArray) -> RealArray:
        # Run Newton's method on the whole real hyperspace.
        return softplus(search_parameters) - 1.0
