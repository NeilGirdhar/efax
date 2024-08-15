from __future__ import annotations

from typing import Any, TypeAlias

from tjax import JaxRealArray
from tjax.dataclasses import dataclass, field
from tjax.fixed_point import ComparingIteratedFunctionWithCombinator, ComparingState
from tjax.gradient import Adam, GradientTransformation
from typing_extensions import override

from .exp_to_nat import ExpToNat, ExpToNatMinimizer

SP: TypeAlias = JaxRealArray


@dataclass
class ExpToNatIteratedFunctionState:
    gradient_transformation_state: Any
    search_parameters: JaxRealArray


@dataclass
class ExpToNatIteratedFunction(
        # The generic parameters are: Parameters, State, Comparand, Differentiand, Trajectory.
        ComparingIteratedFunctionWithCombinator[ExpToNat[Any],
                                                ExpToNatIteratedFunctionState,
                                                SP,
                                                SP,
                                                SP]):
    transform: GradientTransformation[Any, SP] = field()

    @override
    def sampled_state(self,
                      theta: ExpToNat[Any],
                      state: ExpToNatIteratedFunctionState
                      ) -> ExpToNatIteratedFunctionState:
        search_parameters = state.search_parameters
        search_gradient = theta.search_gradient(search_parameters)
        transformed_gradient, new_gt_state = self.transform.update(
                search_gradient, state.gradient_transformation_state, search_parameters)
        new_search_parameters = search_parameters + transformed_gradient
        return ExpToNatIteratedFunctionState(new_gt_state, new_search_parameters)

    @override
    def sampled_state_trajectory(
            self,
            theta: ExpToNat[Any],
            augmented: ComparingState[ExpToNatIteratedFunctionState, SP]
            ) -> tuple[ExpToNatIteratedFunctionState, SP]:
        sampled_state = self.sampled_state(theta, augmented.current_state)
        return sampled_state, sampled_state.search_parameters

    @override
    def extract_comparand(self, state: ExpToNatIteratedFunctionState) -> SP:
        return state.search_parameters

    @override
    def extract_differentiand(self,
                              theta: ExpToNat[Any],
                              state: ExpToNatIteratedFunctionState) -> SP:
        return state.search_parameters

    @override
    def implant_differentiand(self,
                              theta: ExpToNat[Any],
                              state: ExpToNatIteratedFunctionState,
                              differentiand: SP
                              ) -> ExpToNatIteratedFunctionState:
        return ExpToNatIteratedFunctionState(state.gradient_transformation_state, differentiand)


@dataclass
class IteratedMinimizer(ExpToNatMinimizer):
    iterated_function: ExpToNatIteratedFunction

    @override
    def solve(self, exp_to_nat: ExpToNat[Any]) -> SP:
        initial_search_parameters = exp_to_nat.initial_search_parameters()
        initial_gt_state = self.iterated_function.transform.init(initial_search_parameters)
        initial_state = ExpToNatIteratedFunctionState(initial_gt_state, initial_search_parameters)
        augmented_state = self.iterated_function.find_fixed_point(exp_to_nat, initial_state)
        return augmented_state.current_state.search_parameters


default_minimizer = IteratedMinimizer(
        ExpToNatIteratedFunction(minimum_iterations=100,
                                 maximum_iterations=2000,
                                 rtol=0.0,
                                 atol=1e-6,
                                 z_minimum_iterations=100,
                                 z_maximum_iterations=2000,
                                 transform=Adam(5e-2)))
