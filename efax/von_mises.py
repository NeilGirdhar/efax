import math
from typing import Tuple

import numpy as np
import scipy.optimize
from chex import Array
from ipromise import implements
from jax import numpy as jnp
from jax.nn import softplus
from scipy.special import iv
from tjax import RealArray

from .exponential_family import ExponentialFamily

__all__ = ['VonMises', 'VonMisesFisher']


def inverse_softplus(y: Array) -> Array:
    return jnp.where(y > 80.0,
                     y,
                     jnp.log(jnp.expm1(y)))


class VonMisesFisher(ExponentialFamily):

    def __init__(self, num_parameters: int):
        if not isinstance(num_parameters, int):
            raise TypeError
        if num_parameters < 2:
            raise ValueError
        super().__init__(num_parameters=num_parameters,
                         observation_shape=(num_parameters,))

    # Magic methods --------------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"{type(self).__name__}(shape={self.shape}, num_parameters={self.num_parameters})")

    # Implemented methods --------------------------------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealArray) -> RealArray:
        half_k = q.shape[-1] * 0.5
        kappa = jnp.linalg.norm(q, 2, axis=-1)
        return -jnp.log(kappa ** (half_k - 1.0)
                        / ((2.0 * math.pi) ** half_k
                           * iv(half_k - 1.0, kappa)))

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealArray) -> RealArray:
        kappa = jnp.linalg.norm(q, 2, axis=-1, keepdims=True)
        return jnp.where(kappa == 0.0,
                         q,
                         q * (VonMisesFisher._a_k(q.shape[-1], kappa) / kappa))

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealArray) -> RealArray:
        k = p.shape[-1]
        mu = jnp.linalg.norm(p, 2, axis=-1)
        q = np.empty_like(p)
        for i in np.ndindex(p.shape[: -1]):
            this_p = p[i]
            this_mu = mu[i]
            kappa = VonMisesFisher._find_kappa(k, this_mu)
            q[i] = this_p * (kappa / this_mu)
        return q

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealArray) -> RealArray:
        return x

    # Private methods ------------------------------------------------------------------------------
    @staticmethod
    def _a_k(k: RealArray, kappa: RealArray) -> RealArray:
        half_k = k * 0.5
        return iv(half_k, kappa) / iv(half_k - 1.0, kappa)

    @staticmethod
    def _find_kappa(k: RealArray, mu: RealArray) -> RealArray:
        assert 0 <= mu <= 1.0
        if mu == 0.0:
            return 0.0
        initial_solution = (mu * k - mu ** 3) / (1.0 - mu ** 2)

        def f(isp_kappa: RealArray) -> RealArray:
            return VonMisesFisher._a_k(k, softplus(isp_kappa)) - mu
        solution = scipy.optimize.root(f,
                                       inverse_softplus(initial_solution),
                                       tol=1e-5)
        if not solution.success:
            raise ValueError(f"Failed to find kappa because {solution.message}.")
        return softplus(solution.x)


class VonMises(VonMisesFisher):

    def __init__(self) -> None:
        super().__init__(num_parameters=2)

    # Magic methods --------------------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape})"

    # Overridden methods ---------------------------------------------------------------------------
    @staticmethod
    def nat_to_kappa_angle(q: RealArray) -> Tuple[RealArray, RealArray]:
        kappa = np.linalg.norm(q, axis=-1)
        angles = np.where(kappa == 0.0,
                          0.0,
                          np.arctan2(q[..., 1], q[..., 0]))
        return kappa, angles
