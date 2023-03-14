from __future__ import annotations

import jax.numpy as jnp
from tjax import JaxRealArray
from tjax.dataclasses import dataclass

from .dirichlet_common import DirichletCommonEP, DirichletCommonNP

__all__ = ['DirichletNP', 'DirichletEP']


@dataclass
class DirichletNP(DirichletCommonNP['DirichletEP']):
    # Implemented methods --------------------------------------------------------------------------
    def to_exp(self) -> DirichletEP:
        return DirichletEP(self._exp_helper())

    def sufficient_statistics(self, x: JaxRealArray) -> DirichletEP:
        one_minus_total_x = 1.0 - jnp.sum(x, axis=-1, keepdims=True)
        return DirichletEP(jnp.append(jnp.log(x), jnp.log(one_minus_total_x), axis=-1))

    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape[:len(x.shape) - 1])


@dataclass
class DirichletEP(DirichletCommonEP[DirichletNP]):
    # Implemented methods --------------------------------------------------------------------------
    @classmethod
    def natural_parametrization_cls(cls) -> type[DirichletNP]:
        return DirichletNP

    def search_to_natural(self, search_parameters: JaxRealArray) -> DirichletNP:
        return DirichletNP(self._transform_nat_helper(search_parameters))
