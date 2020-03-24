import numpy as np

from efax.tensors import demote_dtype

from ..exponential_family import ExponentialFamily


class DistributionInfo:

    def __init__(self,
                 my_distribution: ExponentialFamily,
                 exp_to_scipy_distribution=None,
                 nat_to_scipy_distribution=None,
                 exp_parameter_generator=None,
                 nat_parameter_generator=None,
                 my_observation=None):
        """
        Args:
            my_distribution:
                a distribution object from this library
            exp_to_scipy_distribution:
                a function that maps expectation parameters to a scipy
                distribution.
            nat_to_scipy_distribution:
                a function that maps natural parameters to a scipy
                distribution.
            exp_parameter_generator:
                generates expectation parameters.  If it's None, it falls back
                to converting values generated by nat_parameter_generator.
            nat_parameter_generator:
                generates natural parameters.  If it's None, it falls back to
                converting values generated by exp_parameter_generator.
            my_observation:
                transforms scipy observations to ones accepted by this
                library.  If it's None, it defaults to the identity function.
        """
        self.my_distribution = my_distribution

        if exp_parameter_generator is not None:
            if isinstance(exp_parameter_generator(np.random, ()), tuple):
                raise TypeError("This should return a number or an ndarray")

        self.exp_to_scipy_distribution = (
            exp_to_scipy_distribution
            if exp_to_scipy_distribution is not None
            else lambda exp: nat_to_scipy_distribution(
                my_distribution.exp_to_nat(exp)))

        self.nat_to_scipy_distribution = (
            nat_to_scipy_distribution
            if nat_to_scipy_distribution is not None
            else lambda nat: exp_to_scipy_distribution(
                my_distribution.nat_to_exp(nat)))

        self._exp_parameter_generator = (
            exp_parameter_generator
            if exp_parameter_generator is not None
            else (lambda rng, shape: my_distribution.nat_to_exp(
                nat_parameter_generator(rng, shape))))

        self._nat_parameter_generator = (
            nat_parameter_generator
            if nat_parameter_generator is not None
            else (lambda rng, shape: my_distribution.exp_to_nat(
                exp_parameter_generator(rng, shape))))

        self.my_observation = ((lambda x: x)
                               if my_observation is None
                               else my_observation)

    # New methods -------------------------------------------------------------
    def exp_parameter_generator(self, rng, shape):
        return demote_dtype(self._exp_parameter_generator(rng, shape))

    def nat_parameter_generator(self, rng, shape):
        return demote_dtype(self._nat_parameter_generator(rng, shape))

    # Magic methods -----------------------------------------------------------
    def __repr__(self):
        return f"DistributionInfo({self.my_distribution})"