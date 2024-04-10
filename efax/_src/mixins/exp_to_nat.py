from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeAlias, TypeVar

import jax.numpy as jnp
from tjax import JaxComplexArray, jit
from tjax.dataclasses import dataclass, field
from tjax.fixed_point import ComparingIteratedFunctionWithCombinator, ComparingState
from tjax.gradient import Adam, GradientTransformation
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import NaturalParametrization
from ..tools import parameter_map

__all__: list[str] = []


NP = TypeVar('NP', bound=NaturalParametrization[Any, Any])
SP: TypeAlias = JaxComplexArray


class ExpToNat(ExpectationParametrization[NP], Generic[NP]):
    """This mixin implements the conversion from expectation to natural parameters.

    It uses Newton's method with a Jacobian to invert the gradient log-normalizer.
    """
    @jit
    @override
    def to_nat(self) -> NP:
        iterated_function = ExpToNatIteratedFunction[NP](minimum_iterations=1000,
                                                         maximum_iterations=1000,
                                                         rtol=1e-4,
                                                         atol=1e-6,
                                                         z_minimum_iterations=100,
                                                         z_maximum_iterations=1000,
                                                         transform=Adam(1e-1))
        initial_search_parameters = self.initial_search_parameters()
        initial_gt_state = iterated_function.transform.init(initial_search_parameters)
        initial_state = ExpToNatIteratedFunctionState(initial_gt_state, initial_search_parameters)
        final_state = iterated_function.find_fixed_point(self, initial_state).current_state
        return self.search_to_natural(final_state.search_parameters)

    def _zero_natural_parameters(self) -> NP:
        """A convenience method for implementing initial_search_parameters."""
        fixed_parameters = self.fixed_parameters()
        cls = type(self).natural_parametrization_cls()
        return cls.unflattened(jnp.zeros_like(self.flattened()), **fixed_parameters)

    @abstractmethod
    def initial_search_parameters(self) -> SP:
        """Returns: The initial value of the parameters that Newton's method runs on."""
        raise NotImplementedError

    @abstractmethod
    def search_to_natural(self, search_parameters: SP) -> NP:
        """Convert the search parameters to the natural parametrization.

        Args:
            search_parameters: The parameters that Newton's method runs on.
        Returns: The corresponding natural parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def search_gradient(self, search_parameters: SP) -> SP:
        raise NotImplementedError

    def _natural_gradient(self, natural_parameters: NP) -> NP:
        """The natural gradient.

        Returns: The difference of the expectation parameters corresponding to natural_parameters
            and self.  This difference is returned as natural parameters.
        """
        expectation_parameters: ExpToNat[NP] = natural_parameters.to_exp()
        exp_difference: ExpToNat[NP] = parameter_map(jnp.subtract, expectation_parameters, self)
        return type(natural_parameters).unflattened(exp_difference.flattened(),
                                                    **self.fixed_parameters())


@dataclass
class ExpToNatIteratedFunctionState:
    gradient_transformation_state: Any
    search_parameters: JaxComplexArray


@dataclass
class ExpToNatIteratedFunction(
        # The generic parameters are: Parameters, State, Comparand, Differentiand, Trajectory.
        ComparingIteratedFunctionWithCombinator[ExpToNat[NP],
                                                ExpToNatIteratedFunctionState,
                                                SP,
                                                SP,
                                                SP],
        Generic[NP]):
    transform: GradientTransformation[Any, SP] = field()

    @override
    def sampled_state(self,
                      theta: ExpToNat[NP],
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
            theta: ExpToNat[NP],
            augmented: ComparingState[ExpToNatIteratedFunctionState, SP]
            ) -> tuple[ExpToNatIteratedFunctionState, SP]:
        sampled_state = self.sampled_state(theta, augmented.current_state)
        return sampled_state, sampled_state.search_parameters

    @override
    def extract_comparand(self, state: ExpToNatIteratedFunctionState) -> SP:
        return state.search_parameters

    @override
    def extract_differentiand(self,
                              theta: ExpToNat[NP],
                              state: ExpToNatIteratedFunctionState) -> SP:
        return state.search_parameters

    @override
    def implant_differentiand(self,
                              theta: ExpToNat[NP],
                              state: ExpToNatIteratedFunctionState,
                              differentiand: SP
                              ) -> ExpToNatIteratedFunctionState:
        return ExpToNatIteratedFunctionState(state.gradient_transformation_state, differentiand)
