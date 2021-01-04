from __future__ import annotations

from typing import Any, Generic, List, TypeVar

from jax import numpy as jnp
from jax.nn import softplus
from jax.scipy import special as jss
from tjax import RealArray, Shape, dataclass

from .exp_to_nat import ExpToNat
from .natural_parametrization import EP, NaturalParametrization
from .parameter import distribution_parameter

__all__: List[str] = []


# TODO: Uncomment this when mypy stop crashing on it.
# EP = TypeVar('EP', bound='DirichletCommonEP[Any]')


@dataclass
class DirichletCommonNP(NaturalParametrization[EP], Generic[EP]):
    alpha_minus_one: RealArray = distribution_parameter(axes=1)

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.alpha_minus_one.shape[:-1]

    def log_normalizer(self) -> RealArray:
        q = self.alpha_minus_one
        return (jnp.sum(jss.gammaln(q + 1.0), axis=-1)
                - jss.gammaln(jnp.sum(q, axis=-1) + q.shape[-1]))

    # Private methods ------------------------------------------------------------------------------
    def _exp_helper(self) -> RealArray:
        q = self.alpha_minus_one
        return jss.digamma(q + 1.0) - jss.digamma(jnp.sum(q, axis=-1, keepdims=True) + q.shape[-1])


NP = TypeVar('NP', bound=DirichletCommonNP[Any])


@dataclass
class DirichletCommonEP(ExpToNat[NP, RealArray], Generic[NP]):
    mean_log_probability: RealArray = distribution_parameter(axes=1)

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean_log_probability.shape[:-1]

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape())

    def initial_search_parameters(self) -> RealArray:
        return jnp.zeros(self.mean_log_probability.shape)

    def search_gradient(self, search_parameters: RealArray) -> RealArray:
        return self._natural_gradient(self.search_to_natural(search_parameters)).alpha_minus_one

    # Private methods ------------------------------------------------------------------------------
    @classmethod
    def _transform_nat_helper(cls, search_parameters: RealArray) -> RealArray:
        # Run Newton's method on the whole real hyperspace.
        return softplus(search_parameters) - 1.0
