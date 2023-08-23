from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from tjax import JaxRealArray
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import NaturalParametrization
from ..parameter import SimplexSupport
from .dirichlet_common import DirichletCommonEP, DirichletCommonNP

__all__ = ['DirichletNP', 'DirichletEP']


@dataclass
class DirichletNP(DirichletCommonNP['DirichletEP'],
                  NaturalParametrization['DirichletEP', JaxRealArray]):
    """The natural parametrization of the Dirichlet distribution.

    Args:
        alpha_minus_one: The shape parameters alpha.
    """
    @override
    @classmethod
    def domain_support(cls) -> SimplexSupport:
        return SimplexSupport()

    @override
    def to_exp(self) -> DirichletEP:
        return DirichletEP(self._exp_helper())

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: Any
                              ) -> DirichletEP:
        one_minus_total_x = 1.0 - jnp.sum(x, axis=-1, keepdims=True)
        return DirichletEP(jnp.append(jnp.log(x), jnp.log(one_minus_total_x), axis=-1))

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        return jnp.zeros(x.shape[:len(x.shape) - 1])


@dataclass
class DirichletEP(DirichletCommonEP[DirichletNP],
                  ExpectationParametrization[DirichletNP]):
    """The expectation parametrization of the Dirichlet distribution.

    Args:
        mean_log_probability: An array with final dimension [E(log(x_i))]_i.
    """
    @override
    @classmethod
    def domain_support(cls) -> SimplexSupport:
        return SimplexSupport()

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[DirichletNP]:
        return DirichletNP

    @override
    def search_to_natural(self, search_parameters: JaxRealArray) -> DirichletNP:
        return DirichletNP(self._transform_nat_helper(search_parameters))
