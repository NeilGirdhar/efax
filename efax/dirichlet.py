from typing import Any

import numpy as np
import scipy.optimize
from ipromise import implements
from jax import numpy as jnp
from jax.scipy import special as jss
from tjax import RealTensor, Shape

from .exponential_family import ExponentialFamily

__all__ = ['Beta', 'Dirichlet']


class Dirichlet(ExponentialFamily):

    def __init__(self, num_parameters: int, **kwargs: Any):
        if not isinstance(num_parameters, int):
            raise TypeError
        if num_parameters < 2:
            raise ValueError
        observation_shape: Shape
        if num_parameters == 2:
            observation_shape = ()
        else:
            observation_shape = (num_parameters - 1,)
        super().__init__(num_parameters=num_parameters,
                         observation_shape=observation_shape,
                         **kwargs)

    # Magic methods --------------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"{type(self).__name__}(shape={self.shape}, "
                f"num_parameters={self.num_parameters})")

    # Implemented methods --------------------------------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealTensor) -> RealTensor:
        return (jnp.sum(jss.gammaln(q + 1.0), axis=-1)
                - jss.gammaln(jnp.sum(q, axis=-1) + q.shape[-1]))

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealTensor) -> RealTensor:
        return (jss.digamma(q + 1.0)
                - jss.digamma(jnp.sum(q, axis=-1, keepdims=True) + q.shape[-1]))

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealTensor) -> RealTensor:
        q = np.empty_like(p)
        for i in np.ndindex(p.shape[: -1]):
            this_p = p[i]

            def f(some_q: RealTensor,
                  this_p: RealTensor = this_p) -> RealTensor:
                some_q = jnp.maximum(-0.999, some_q)
                some_p = self.nat_to_exp(some_q)
                return some_p - this_p

            solution = scipy.optimize.root(f,
                                           np.zeros_like(this_p),
                                           tol=1e-5)
            if not solution.success:
                raise ValueError(
                    "Failed to find natural parmaeters for "
                    f"{this_p} because {solution.message}.")
            q[i] = solution.x
        return q

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealTensor) -> RealTensor:
        return jnp.log(x)


class Beta(Dirichlet):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(num_parameters=2, **kwargs)
