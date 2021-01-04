from __future__ import annotations

from typing import Any, Generic, Type, TypeVar

from jax import numpy as jnp
from jax.nn import softplus
from jax.scipy import special as jss
from tjax import RealArray, Shape, dataclass

from .exp_to_nat import ExpToNat
from .natural_parametrization import EP, NaturalParametrization
from .parameter import distribution_parameter

__all__ = ['DirichletNP', 'DirichletEP']


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

@dataclass
class DirichletNP(DirichletCommonNP['DirichletEP']):
    # Implemented methods --------------------------------------------------------------------------
    def to_exp(self) -> DirichletEP:
        return DirichletEP(self._exp_helper())

    def sufficient_statistics(self, x: RealArray) -> DirichletEP:
        one_minus_total_x = 1.0 - jnp.sum(x, axis=-1, keepdims=True)
        return DirichletEP(jnp.append(jnp.log(x), jnp.log(one_minus_total_x), axis=-1))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape[:len(x.shape) - 1])


NP = TypeVar('NP', bound=DirichletCommonNP[Any])


@dataclass
class DirichletCommonEP(ExpToNat[NP], Generic[NP]):
    mean_log_probability: RealArray = distribution_parameter(axes=1)

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean_log_probability.shape[:-1]

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape())

    # Private methods ------------------------------------------------------------------------------
    @classmethod
    def _transform_nat_helper(cls, iteration_natural: NP) -> RealArray:
        # Run Newton's method on the whole real hyperspace.
        return softplus(iteration_natural.alpha_minus_one) - 1.0


@dataclass
class DirichletEP(DirichletCommonEP[DirichletNP]):
    # Implemented methods --------------------------------------------------------------------------
    @classmethod
    def natural_parametrization_cls(cls) -> Type[DirichletNP]:
        return DirichletNP

    @classmethod
    def transform_natural_for_iteration(cls, iteration_natural: DirichletNP) -> DirichletNP:
        return DirichletNP(cls._transform_nat_helper(iteration_natural))
