from __future__ import annotations

from typing import Any, Generic, List, Tuple, TypeVar

from jax import jit
from jax import numpy as jnp
from jax.tree_util import tree_multimap
from tjax import dataclass, field_names_and_values
from tjax.fixed_point import ComparingIteratedFunctionWithCombinator, ComparingState
from tjax.gradient import GradientTransformation, adam

from .expectation_parametrization import ExpectationParametrization
from .natural_parametrization import NaturalParametrization

__all__: List[str] = []


NP = TypeVar('NP', bound=NaturalParametrization[Any])


class ExpToNat(ExpectationParametrization[NP], Generic[NP]):
    """
    This mixin implements the conversion from expectation to natural parameters using Newton's
    method with a Jacobian to invert the gradient log-normalizer.
    """

    # Implemented methods --------------------------------------------------------------------------
    @jit
    def to_nat(self) -> NP:
        iterated_function = ExpToNatIteratedFunction[NP](minimum_iterations=1000,
                                                         maximum_iterations=1000,
                                                         transform=adam(1e-1))
        initial_natural = self.initial_natural()
        initial_gt_state = iterated_function.transform.init(initial_natural)
        initial_state = initial_gt_state, initial_natural
        final_state = iterated_function.find_fixed_point(self, initial_state).current_state
        _, final_natural = final_state
        return self.transform_natural_for_iteration(final_natural)

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
                                                Tuple[Any, NP],
                                                ExpToNat[NP],
                                                NP,
                                                NP],
        Generic[NP]):

    transform: GradientTransformation[Any, NP]

    def sampled_state(self, theta: ExpToNat[NP], state: Tuple[Any, NP]) -> Tuple[Any, NP]:
        current_gt_state, untransformed_natural_parameters = state
        natural_parameters = theta.transform_natural_for_iteration(untransformed_natural_parameters)
        expectation_parameters = natural_parameters.to_exp()
        exp_difference = tree_multimap(jnp.subtract, theta, expectation_parameters)
        gradient = type(natural_parameters).unflattened(-exp_difference.flattened())

        ##gradient = id_print(gradient, what="G")
        transformed_gradient, new_gt_state = self.transform.update(gradient, current_gt_state,
                                                                   untransformed_natural_parameters)
        #transformed_gradient = id_print(transformed_gradient, what="T")
        new_untransformed_natural_parameters = tree_multimap(jnp.add,
                                                             untransformed_natural_parameters,
                                                             transformed_gradient)
        return new_gt_state, new_untransformed_natural_parameters

    def sampled_state_trajectory(
            self,
            theta: ExpToNat[NP],
            augmented: ComparingState[Tuple[Any, NP], ExpToNat[NP]]) -> Tuple[Tuple[Any, NP], NP]:
        sampled_state = self.sampled_state(theta, augmented.current_state)
        _, trajectory = sampled_state
        return sampled_state, trajectory

    def extract_comparand(self, state: Tuple[Any, NP]) -> ExpToNat[NP]:
        _, untransformed_natural_parameters = state
        return untransformed_natural_parameters.to_exp()

    def extract_differentiand(self, state: Tuple[Any, NP]) -> NP:
        _, untransformed_natural_parameters = state
        return untransformed_natural_parameters

    def implant_differentiand(self,
                              state: Tuple[Any, NP],
                              differentiand: NP) -> Tuple[Any, NP]:
        current_gt_state, _ = state
        return current_gt_state, differentiand
