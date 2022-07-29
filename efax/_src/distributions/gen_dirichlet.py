"""
The generalized Dirichlet distribution is based off:

    T.-T. Wong 1998. Generalized Dirichlet distribution in Bayesian analysis. Applied Mathematics and Computation, volume 97, pp165-181
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import grad, vmap
from jax.nn import softplus
from tjax import RealArray, Shape
from tjax.dataclasses import dataclass

from ..exp_to_nat import ExpToNat
from ..multidimensional import Multidimensional
from ..natural_parametrization import NaturalParametrization
from ..parameter import VectorSupport, distribution_parameter
from ..tools import betaln

__all__ = ['GeneralizedDirichletNP', 'GeneralizedDirichletEP']


@dataclass
class GeneralizedDirichletNP(NaturalParametrization['GeneralizedDirichletEP', RealArray],
                             Multidimensional):
    alpha_minus_one: RealArray = distribution_parameter(VectorSupport())
    gamma: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.alpha_minus_one.shape[:-1]

    def log_normalizer(self) -> RealArray:
        alpha, beta = self.alpha_beta()
        return jnp.sum(betaln(alpha, beta), axis=-1)

    def to_exp(self) -> GeneralizedDirichletEP:
        def f(gen_np: GeneralizedDirichletNP) -> RealArray:
            alpha, beta = gen_np.alpha_beta()
            return jnp.sum(betaln(alpha, beta), axis=-1)
        f = grad(f)
        for _ in range(len(self.shape)):
            f = vmap(f)
        values = f(self)
        flattened = values.flattened()
        return GeneralizedDirichletEP.unflattened(flattened)

    def sufficient_statistics(self, x: RealArray) -> GeneralizedDirichletEP:
        cs_x = jnp.cumsum(x, axis=-1)
        return GeneralizedDirichletEP(jnp.log(x), jnp.log(1.0 - cs_x))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape[:len(x.shape) - 1])

    def dimensions(self) -> int:
        return self.alpha_minus_one.shape[-1]

    # New methods ----------------------------------------------------------------------------------
    def alpha_beta(self) -> tuple[RealArray, RealArray]:
        alpha = self.alpha_minus_one + 1.0
        # cs_alpha[i] = sum_{j>=i} alpha[j]
        # cs_gamma[i] = sum_{j>=i} gamma[j]
        cs_alpha = jnp.cumsum(alpha[..., ::-1], axis=-1)[..., ::-1]
        cs_gamma = jnp.cumsum(self.gamma[..., ::-1], axis=-1)[..., ::-1]
        # roll_cs_alpha[i] = sum_{j>i} cs_alpha[j]
        roll_cs_alpha = jnp.roll(cs_alpha, -1, axis=-1)
        roll_cs_alpha = roll_cs_alpha.at[..., -1].set(0.0)
        beta = cs_gamma + roll_cs_alpha + 1.0
        return alpha, beta


@dataclass
class GeneralizedDirichletEP(ExpToNat[GeneralizedDirichletNP, RealArray], Multidimensional):
    mean_log_probability: RealArray = distribution_parameter(VectorSupport())
    mean_log_cumulative_probability: RealArray = distribution_parameter(VectorSupport())

    # Implemented methods --------------------------------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self.mean_log_probability.shape[:-1]

    @classmethod
    def natural_parametrization_cls(cls) -> type[GeneralizedDirichletNP]:
        return GeneralizedDirichletNP

    def search_to_natural(self, search_parameters: RealArray) -> GeneralizedDirichletNP:
        # Run Newton's method on the whole real hyperspace.
        n = self.dimensions()
        positive_search_parameters = softplus(search_parameters)
        return GeneralizedDirichletNP(positive_search_parameters[..., :n] - 1.0,
                                      positive_search_parameters[..., n:])

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape)

    def initial_search_parameters(self) -> RealArray:
        return jnp.zeros(self.shape + (self.dimensions() * 2,))

    def search_gradient(self, search_parameters: RealArray) -> RealArray:
        return self._natural_gradient(self.search_to_natural(search_parameters)).flattened()

    def dimensions(self) -> int:
        return self.mean_log_probability.shape[-1]
