from typing import Optional

import numpy as np
from numpy.random import Generator
from scipy import stats as ss
from tjax import Shape

__all__ = ['ScipyMultivariateNormal']


# pylint: disable=protected-access
class ScipyMultivariateNormal(ss._multivariate.multivariate_normal_frozen):

    def rvs(self,
            size: Optional[Shape] = None,
            generator: Optional[Generator] = None) -> np.ndarray:
        retval = super().rvs(size=size, random_state=generator)
        if size == (1,):
            retval = retval.reshape((1, 2))
        return retval
