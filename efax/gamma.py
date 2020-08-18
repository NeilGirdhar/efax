import math

import numpy as np
import scipy
from ipromise import implements
from jax import numpy as jnp
from jax.scipy import special as jss
from scipy.special import polygamma
from tjax import RealArray

from .exponential_family import ExponentialFamily

__all__ = ['Gamma']


def trigamma(x: RealArray) -> RealArray:
    return polygamma(1, x)


class Gamma(ExponentialFamily):

    def __init__(self) -> None:
        super().__init__(num_parameters=2)

    # Implemented methods --------------------------------------------------------------------------
    @implements(ExponentialFamily)
    def log_normalizer(self, q: RealArray) -> RealArray:
        negative_rate = q[..., 0]
        shape_minus_one = q[..., 1]
        shape = shape_minus_one + 1.0
        return jss.gammaln(shape) - shape * jnp.log(-negative_rate)

    @implements(ExponentialFamily)
    def nat_to_exp(self, q: RealArray) -> RealArray:
        negative_rate = q[..., 0]
        shape_minus_one = q[..., 1]
        shape = shape_minus_one + 1.0
        return jnp.stack([-shape / negative_rate,
                          jss.digamma(shape) - jnp.log(-negative_rate)],
                         axis=-1)

    @implements(ExponentialFamily)
    def exp_to_nat(self, p: RealArray) -> RealArray:
        mean = p[..., 0]
        mean_log = p[..., 1]
        shape = Gamma.solve_for_shape(mean, mean_log)
        rate = shape / mean
        return jnp.stack([-rate, shape - 1.0], axis=-1)

    @implements(ExponentialFamily)
    def sufficient_statistics(self, x: RealArray) -> RealArray:
        return jnp.stack([x, jnp.log(x)], axis=-1)

    # New methods ----------------------------------------------------------------------------------
    @staticmethod
    def solve_for_shape(mean: RealArray, mean_log: RealArray) -> RealArray:
        def f(shape: float) -> float:
            return math.log(shape) - scipy.special.digamma(shape) - log_mean_minus_mean_log

        def f_prime(shape: float) -> float:
            return 1.0 / shape - trigamma(shape)

        output_shape = np.empty_like(mean)
        it = np.nditer([mean, mean_log, output_shape],
                       op_flags=[['readonly'], ['readonly'], ['writeonly', 'allocate']])

        with it:
            for this_mean, this_mean_log, this_shape in it:
                log_mean_minus_mean_log = math.log(this_mean) - this_mean_log
                initial_shape = ((3.0
                                  - log_mean_minus_mean_log
                                  + math.sqrt((log_mean_minus_mean_log - 3.0) ** 2
                                              + 24.0 * log_mean_minus_mean_log))
                                 / (12.0 * log_mean_minus_mean_log))

                this_shape[...] = scipy.optimize.newton(f, initial_shape, fprime=f_prime)
        return output_shape

    @staticmethod
    def solve_for_shape_and_scale(mean: RealArray, mean_log: RealArray) -> RealArray:
        shape = Gamma.solve_for_shape(mean, mean_log)
        scale = mean / shape
        return shape, scale
