from __future__ import annotations

from jax import numpy as jnp
from jax.nn import softplus
from jax.scipy import special as jss
from tjax import RealArray, Shape, dataclass

from .exp_to_nat import ExpToNat
from .exponential_family import NaturalParametrization
from .parameter import distribution_parameter

__all__ = ['DirichletNP', 'DirichletEP']


@dataclass
class DirichletNP(NaturalParametrization['DirichletEP']):
    alpha_minus_one: RealArray = distribution_parameter(axes=1)

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.alpha_minus_one.shape[:-1]

    def log_normalizer(self) -> RealArray:
        q = self.alpha_minus_one
        return (jnp.sum(jss.gammaln(q + 1.0), axis=-1)
                - jss.gammaln(jnp.sum(q, axis=-1) + q.shape[-1]))

    def to_exp(self) -> DirichletEP:
        q = self.alpha_minus_one
        return DirichletEP(jss.digamma(q + 1.0)
                           - jss.digamma(jnp.sum(q, axis=-1, keepdims=True) + q.shape[-1]))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape[:len(x.shape) - len(self.observation_shape())])

    def sufficient_statistics(self, x: RealArray) -> DirichletEP:
        if self.observation_shape() == ():
            return DirichletEP(jnp.stack([jnp.log(x), jnp.log(1.0 - x)], axis=-1))
        one_minus_total_x = 1.0 - jnp.sum(x, axis=-1, keepdims=True)
        return DirichletEP(jnp.append(jnp.log(x), jnp.log(one_minus_total_x), axis=-1))

    # New methods ----------------------------------------------------------------------------------
    def observation_shape(self) -> Shape:
        if self.alpha_minus_one.shape[-1] <= 1:
            raise ValueError
        if self.alpha_minus_one.shape[-1] == 2:
            return ()
        return (self.alpha_minus_one.shape[-1],)


@dataclass
class DirichletEP(ExpToNat[DirichletNP]):
    mean_log_probability: RealArray = distribution_parameter(axes=1)

    # Implemented methods --------------------------------------------------------------------------
    @classmethod
    def natural_parametrization_cls(cls) -> Type[DirichletNP]:
        return DirichletNP

    def shape(self) -> Shape:
        return self.mean_log_probability.shape[:-1]

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape())

    @classmethod
    def transform_natural_for_iteration(cls, iteration_natural: DirichletNP) -> DirichletNP:
        # Run Newton's method on the whole real line.
        return DirichletNP(softplus(iteration_natural.alpha_minus_one) - 1.0)
