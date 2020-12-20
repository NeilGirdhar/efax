from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import scipy
from jax import numpy as jnp
from jax.scipy import special as jss
from scipy.special import polygamma
from tjax import RealArray, Shape, dataclass

from .exponential_family import ExpectationParametrization, NaturalParametrization

__all__ = ['GammaNP', 'GammaEP']


@dataclass
class GammaNP(NaturalParametrization['GammaEP']):
    negative_rate: RealArray
    shape_minus_one: RealArray

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.negative_rate.shape

    def log_normalizer(self) -> RealArray:
        shape = self.shape_minus_one + 1.0
        return jss.gammaln(shape) - shape * jnp.log(-self.negative_rate)

    def to_exp(self) -> GammaEP:
        shape = self.shape_minus_one + 1.0
        return GammaEP(-shape / self.negative_rate,
                       jss.digamma(shape) - jnp.log(-self.negative_rate))

    def carrier_measure(self, x: RealArray) -> RealArray:
        return jnp.zeros(x.shape)

    def sufficient_statistics(self, x: RealArray) -> GammaEP:
        return GammaEP(x, jnp.log(x))

    @classmethod
    def field_axes(cls) -> Iterable[int]:
        yield 0
        yield 0


@dataclass
class GammaEP(ExpectationParametrization[GammaNP]):
    mean: RealArray
    mean_log: RealArray  # digamma(k) - log(rate)

    # Implemented methods --------------------------------------------------------------------------
    def shape(self) -> Shape:
        return self.mean.shape

    def to_nat(self) -> GammaNP:
        shape = self.solve_for_shape()
        rate = shape / self.mean
        return GammaNP(-rate, shape - 1.0)

    def expected_carrier_measure(self) -> RealArray:
        return jnp.zeros(self.shape())

    # New methods ----------------------------------------------------------------------------------
    def solve_for_shape(self) -> RealArray:
        def f(shape: float) -> float:
            return math.log(shape) - scipy.special.digamma(shape) - log_mean_minus_mean_log

        def f_prime(shape: float) -> float:
            return 1.0 / shape - polygamma(1, shape)  # polygamma(1) is trigamma

        output_shape = np.empty_like(self.mean)
        it = np.nditer([self.mean, self.mean_log, output_shape],
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

    def solve_for_shape_and_scale(self) -> RealArray:
        shape = self.solve_for_shape()
        scale = self.mean / shape
        return shape, scale
