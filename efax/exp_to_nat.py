from __future__ import annotations

from typing import Any, Generic, List, Tuple, TypeVar

import jax.numpy as jnp
from jax import jit
from jax.tree_util import tree_multimap
from tjax import dataclass, field_names_and_values
from tjax.fixed_point import ComparingIteratedFunctionWithCombinator, ComparingState
from tjax.gradient import GradientTransformation, adam

from .expectation_parametrization import ExpectationParametrization
from .natural_parametrization import NaturalParametrization

__all__: List[str] = []


NP = TypeVar('NP', bound=NaturalParametrization[Any])
SP = TypeVar('SP', bound=Any)


class ExpToNat(ExpectationParametrization[NP], Generic[NP, SP]):
    """
    This mixin implements the conversion from expectation to natural parameters using Newton's
    method with a Jacobian to invert the gradient log-normalizer.
    """

    # Implemented methods --------------------------------------------------------------------------
    @jit
    def to_nat(self) -> NP:
        iterated_function = ExpToNatIteratedFunction[NP, SP](minimum_iterations=1000,
                                                             maximum_iterations=1000,
                                                             transform=adam(1e-1))
        initial_search_parameters = self.initial_search_parameters()
        initial_gt_state = iterated_function.transform.init(initial_search_parameters)
        initial_state = initial_gt_state, initial_search_parameters
        final_state = iterated_function.find_fixed_point(self, initial_state).current_state
        _, final_search_parameters = final_state
        return self.search_to_natural(final_search_parameters)

    # Non-final methods ----------------------------------------------------------------------------
    def _zero_natural_parameters(self) -> NP:
        "A convenience method for implementing initial_search_parameters"
        kwargs = dict(field_names_and_values(self, static=True))
        cls = type(self).natural_parametrization_cls()
        return cls.unflattened(jnp.zeros_like(self.flattened()), **kwargs)

    # Abstract methods -----------------------------------------------------------------------------
    def initial_search_parameters(self) -> SP:
        """
        Returns: The initial value of the parameters that Newton's method runs on.
        """
        raise NotImplementedError

    def search_to_natural(self, search_parameters: SP) -> NP:
        """
        Args:
            search_parameters: The parameters that Newton's method runs on.
        Returns: The corresponding natural parametrization.
        """
        raise NotImplementedError

    def search_gradient(self, search_parameters: SP) -> SP:
        raise NotImplementedError

    # Private methods ------------------------------------------------------------------------------
    def _natural_gradient(self, natural_parameters: NP) -> NP:
        """
        Returns: The difference of the expectation parameters corresponding to natural_parameters
            and self.  This difference is returned as natural parameters.
        """
        expectation_parameters: ExpToNat[NP, SP] = natural_parameters.to_exp()
        exp_difference: ExpToNat[NP, SP] = tree_multimap(jnp.subtract, expectation_parameters, self)
        return type(natural_parameters).unflattened(exp_difference.flattened())


@dataclass
class ExpToNatIteratedFunction(
        # The generic parameters are: Parameters, State, Comparand, Differentiand, Trajectory.
        ComparingIteratedFunctionWithCombinator[ExpToNat[NP, SP],
                                                Tuple[Any, SP],
                                                SP,
                                                SP,
                                                NP],
        Generic[NP, SP]):

    transform: GradientTransformation[Any, NP]

    def sampled_state(self, theta: ExpToNat[NP, SP], state: Tuple[Any, SP]) -> Tuple[Any, SP]:
        current_gt_state, search_parameters = state
        search_gradient = theta.search_gradient(search_parameters)
        transformed_gradient, new_gt_state = self.transform.update(search_gradient,
                                                                   current_gt_state,
                                                                   search_parameters)
        new_search_parameters = tree_multimap(jnp.add, search_parameters, transformed_gradient)
        return new_gt_state, new_search_parameters

    def sampled_state_trajectory(
            self,
            theta: ExpToNat[NP, SP],
            augmented: ComparingState[Tuple[Any, SP], SP]) -> Tuple[Tuple[Any, SP], NP]:
        sampled_state = self.sampled_state(theta, augmented.current_state)
        _, trajectory = sampled_state
        return sampled_state, trajectory

    def extract_comparand(self, state: Tuple[Any, SP]) -> SP:
        _, search_parameters = state
        return search_parameters

    def extract_differentiand(self, state: Tuple[Any, SP]) -> SP:
        _, search_parameters = state
        return search_parameters

    def implant_differentiand(self,
                              state: Tuple[Any, SP],
                              differentiand: SP) -> Tuple[Any, SP]:
        current_gt_state, _ = state
        return current_gt_state, differentiand
