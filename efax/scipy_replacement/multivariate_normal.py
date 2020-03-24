import scipy.stats as ss

__all__ = ['ScipyMultivariateNormal']


# pylint: disable=protected-access
class ScipyMultivariateNormal(ss._multivariate.multivariate_normal_frozen):

    def rvs(self, size=None, generator=None):
        retval = super().rvs(size=size, random_state=generator)
        if size == (1,):
            retval = retval.reshape((1, 2))
        return retval
