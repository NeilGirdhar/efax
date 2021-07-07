from typing import Optional

from numpy.random import Generator
from scipy import stats as ss
from tjax import RealArray, ShapeLike

__all__ = ['ScipyMultivariateNormal']


# pylint: disable=protected-access
class ScipyMultivariateNormal(ss._multivariate.multivariate_normal_frozen):
    def rvs(self,
            size: Optional[ShapeLike] = None,
            random_state: Optional[Generator] = None) -> RealArray:
        retval = super().rvs(size=size, random_state=random_state)
        if size is None:
            size = ()
        elif isinstance(size, int):
            size = (size,)
        else:
            size = tuple(size)
        return retval.reshape(size + self.mean.shape)
