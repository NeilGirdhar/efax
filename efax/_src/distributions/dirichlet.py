from __future__ import annotations

import jax.numpy as jnp
from tjax import JaxRealArray
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..parameter import SimplexSupport
from .dirichlet_common import DirichletCommonEP, DirichletCommonNP

__all__ = ['DirichletNP', 'DirichletEP']


@dataclass
class DirichletNP(DirichletCommonNP['DirichletEP']):
    @override
    def domain_support(self) -> SimplexSupport:
        return SimplexSupport()

    @override
    def to_exp(self) -> DirichletEP:
        return DirichletEP(self._exp_helper())

    @override
    def sufficient_statistics(self, x: JaxRealArray) -> DirichletEP:
        one_minus_total_x = 1.0 - jnp.sum(x, axis=-1, keepdims=True)
        return DirichletEP(jnp.append(jnp.log(x), jnp.log(one_minus_total_x), axis=-1))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape[:len(x.shape) - 1])


@dataclass
class DirichletEP(DirichletCommonEP[DirichletNP]):
    @override
    def domain_support(self) -> SimplexSupport:
        return SimplexSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[DirichletNP]:
        return DirichletNP

    @override
    def search_to_natural(self, search_parameters: JaxRealArray) -> DirichletNP:
        return DirichletNP(self._transform_nat_helper(search_parameters))
