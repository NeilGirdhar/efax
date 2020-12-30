from __future__ import annotations

from typing import Any, Generic, List, Optional, Tuple, TypeVar

from chex import Array
from jax import numpy as jnp
from jax.tree_util import tree_multimap
from tjax import field_names_and_values, dataclass
from tjax.fixed_point import ComparingIteratedFunctionWithCombinator, ComparingState

from .exponential_family import ExpectationParametrization, NaturalParametrization

__all__: List[str] = []


NP = TypeVar('NP', bound=NaturalParametrization[Any])


class ExpToNat(ExpectationParametrization[NP], Generic[NP]):
    """
    This mixin implements the conversion from expectation to natural parameters using Newton's
    method with a Jacobian to invert the gradient log-normalizer.
    """

    # Implemented methods --------------------------------------------------------------------------
    def to_nat(self) -> NP:
        iterated_function = ExpToNatIteratedFunction[NP](minimum_iterations=1000,
                                                         maximum_iterations=1000)
        state = iterated_function.find_fixed_point(self, self.initial_natural()).current_state
        return self.transform_natural_for_iteration(state)

    # Non-final methods ----------------------------------------------------------------------------
    def initial_natural(self) -> NP:
        """
        Returns: A form of the natural parameters that Newton's method will run on.
        """
        kwargs = dict(field_names_and_values(self, static=True))
        cls = type(self).natural_parametrization_cls()
        return cls.unflattened(jnp.zeros_like(self.flattened()), **kwargs)

    @classmethod
    def transform_natural_for_iteration(cls, iteration_natural: NP) -> NP:
        """
        Args:
            iteration_natural: A form of the natural parameters that Newton's method will run on.
        Returns: The corresponding natural parametrization.
        """
        raise NotImplementedError


@dataclass
class ExpToNatIteratedFunction(
        # The generic parameters are: Parameters, State, Comparand, Differentiand, Trajectory.
        ComparingIteratedFunctionWithCombinator[ExpToNat[NP],
                                                NP,
                                                ExpToNat[NP],
                                                NP,
                                                NP],
        Generic[NP]):

    def sampled_state(self, theta: ExpToNat[NP], state: NP) -> NP:
        def scaled_difference(target: Array, guess: Array) -> Array:
            return 1e-1 * (guess - target)

        def delta(theta: ExpToNat[NP], state: NP) -> NP:
            natural_parameters = theta.transform_natural_for_iteration(state)
            expectation_parameters = natural_parameters.to_exp()
            exp_difference = tree_multimap(scaled_difference, theta, expectation_parameters)
            nat_cls = type(state)
            return nat_cls.unflattened(exp_difference.flattened())
        # TODO: use jacobian

        # Adjust state in the direction of delta.
        the_delta = delta(theta, state)
        return tree_multimap(jnp.subtract, state, the_delta)

    def sampled_state_trajectory(self,
                                 theta: ExpToNat[NP],
                                 augmented: ComparingState[NP, ExpToNat[NP]]) -> Tuple[NP, NP]:
        sampled_state = self.sampled_state(theta, augmented.current_state)
        trajectory = sampled_state
        return sampled_state, trajectory

    def extract_comparand(self, state: NP) -> ExpToNat[NP]:
        return state.to_exp()

    def extract_differentiand(self, state: NP) -> NP:
        return state

    def implant_differentiand(self, state: NP, differentiand: NP) -> NP:
        return differentiand
