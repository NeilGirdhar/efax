from typing import Optional

from numpy.random import Generator
from scipy import stats as ss
from tjax import RealArray, Shape

__all__ = ['ScipyMultivariateNormal']


# pylint: disable=protected-access
class ScipyMultivariateNormal(ss._multivariate.multivariate_normal_frozen):
    def rvs(self,
            size: Optional[Shape] = None,
            random_state: Optional[Generator] = None) -> RealArray:
        retval = super().rvs(size=size, random_state=random_state)
        if size == (1,):
            assert False
            retval = retval.reshape((1, 2))
        return retval
