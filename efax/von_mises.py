import math

import numpy as np
import scipy.optimize
from ipromise import implements
from jax import numpy as jnp
from jax.nn import softplus
from scipy.special import iv

from cmm.tensors import inverse_softplus

from .exponential_family import ExponentialFamily

__all__ = ['VonMises', 'VonMisesFisher']


class VonMisesFisher(ExponentialFamily):

    def __init__(self, num_parameters):
        if not isinstance(num_parameters, int):
            raise TypeError
        if num_parameters < 2:
            raise ValueError
        super().__init__(num_parameters=num_parameters)

    # Magic methods -----------------------------------------------------------
    def __repr__(self):
        return (f"{type(self).__name__}(shape={self.shape}, "
                f"num_parameters={self.num_parameters})")

    # Implemented methods -----------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q):
        half_k = q.shape[-1] * 0.5
        kappa = jnp.linalg.norm(q, 2, axis=-1)
        return -jnp.log(kappa ** (half_k - 1.0)
                        / ((2.0 * math.pi) ** half_k
                           * iv(half_k - 1.0, kappa)))

    @implements(ExponentialFamily)
    def nat_to_exp(self, q):
        kappa = jnp.linalg.norm(q, 2, axis=-1, keepdims=True)
        return jnp.where(kappa == 0.0,
                         q,
                         q * (VonMisesFisher._a_k(q.shape[-1], kappa) / kappa))

    @implements(ExponentialFamily)
    def exp_to_nat(self, p):
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
    def sufficient_statistics(self, x):
        return x

    # Private methods ---------------------------------------------------------
    @staticmethod
    def _a_k(k, kappa):
        half_k = k * 0.5
        return iv(half_k, kappa) / iv(half_k - 1.0, kappa)

    @staticmethod
    def _find_kappa(k, mu):
        assert 0 <= mu <= 1.0
        if mu == 0.0:
            return 0.0
        initial_solution = (mu * k - mu ** 3) / (1.0 - mu ** 2)

        def f(isp_kappa):
            return VonMisesFisher._a_k(k, softplus(isp_kappa)) - mu
        solution = scipy.optimize.root(f,
                                       inverse_softplus(initial_solution),
                                       tol=1e-5)
        if not solution.success:
            raise ValueError(
                f"Failed to find kappa for because {solution.message}.")
        return softplus(solution.x)


class VonMises(VonMisesFisher):

    def __init__(self):
        super().__init__(num_parameters=2)

    # Magic methods -----------------------------------------------------------
    def __repr__(self):
        return f"{type(self).__name__}(shape={self.shape})"

    # Overridden methods ------------------------------------------------------
    @staticmethod
    def nat_to_kappa_angle(q):
        kappa = np.linalg.norm(q, axis=-1)
        angles = np.where(kappa == 0.0,
                          0.0,
                          np.arctan2(q[..., 1], q[..., 0]))
        return kappa, angles
