from __future__ import annotations

from functools import partial
from typing import Any, Generic, List, TypeVar

from chex import Array
from jax import jit
from jax import numpy as jnp
from jax import value_and_grad, vmap
from jax.tree_util import tree_multimap
from tensorflow_probability.substrates.jax.optimizer import lbfgs_minimize
from tjax import field_names_and_values

from .exponential_family import ExpectationParametrization, NaturalParametrization, tree_dot_final

__all__: List[str] = []


NP = TypeVar('NP', bound=NaturalParametrization[Any])


class ExpToNat(ExpectationParametrization[NP], Generic[NP]):
    """
    This mixin implements the conversion from expectation to natural parameters using Newton's
    method with a Jacobian to invert the gradient log-normalizer.
    """

    # Implemented methods --------------------------------------------------------------------------
    def to_nat(self) -> NP:
        nat_cls = self.natural_parametrization_cls()

        @jit
        def f(target: ExpToNat[NP], x: Array) -> Array:
            state = nat_cls.unflattened(x)
            natural_parameters = target.transform_natural_for_iteration(state)
            return -tree_dot_final(natural_parameters, self) + natural_parameters.log_normalizer()
            expectation_parameters = natural_parameters.to_exp()
            exp_difference = tree_multimap(jnp.subtract, expectation_parameters, target)
            retval = jnp.sum(jnp.square(exp_difference.flattened()), axis=-1)
            return retval

        f = vmap((value_and_grad(f, argnums=1)))
        fp = partial(f, self)

        results = lbfgs_minimize(fp, self.initial_natural().flattened(),
                                 tolerance=1e-5)
        natural_parameters = nat_cls.unflattened(results.position)
        return self.transform_natural_for_iteration(natural_parameters)

    # Non-final methods ----------------------------------------------------------------------------
    def initial_natural(self) -> NP:
        """
        Returns: A form of the natural parameters that Newton's method will run on.
        """
        kwargs = dict(field_names_and_values(self, static=True))
        nat_cls = type(self).natural_parametrization_cls()
        return nat_cls.unflattened(jnp.zeros_like(self.flattened()), **kwargs)

    @classmethod
    def transform_natural_for_iteration(cls, iteration_natural: NP) -> NP:
        """
        Args:
            iteration_natural: A form of the natural parameters that Newton's method will run on.
        Returns: The corresponding natural parametrization.
        """
        raise NotImplementedError
