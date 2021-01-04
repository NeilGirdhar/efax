from __future__ import annotations

from typing import Type

from jax import numpy as jnp
from tjax import RealArray, dataclass

from .dirichlet_common import DirichletCommonEP, DirichletCommonNP

__all__ = ['DirichletNP', 'DirichletEP']


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


@dataclass
class DirichletEP(DirichletCommonEP[DirichletNP]):
    # Implemented methods --------------------------------------------------------------------------
    @classmethod
    def natural_parametrization_cls(cls) -> Type[DirichletNP]:
        return DirichletNP

    @classmethod
    def transform_natural_for_iteration(cls, iteration_natural: DirichletNP) -> DirichletNP:
        return DirichletNP(cls._transform_nat_helper(iteration_natural))
