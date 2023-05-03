from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

import jax.numpy as jnp
from jax.tree_util import tree_map
from tjax import jit
from tjax.dataclasses import dataclass, field
from tjax.fixed_point import ComparingIteratedFunctionWithCombinator, ComparingState
from tjax.gradient import Adam, GradientTransformation
from typing_extensions import override

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import NaturalParametrization

__all__: list[str] = []


NP = TypeVar('NP', bound=NaturalParametrization[Any, Any])
SP = TypeVar('SP', bound=Any)


class ExpToNat(ExpectationParametrization[NP], Generic[NP, SP]):
    """This mixin implements the conversion from expectation to natural parameters.

    It uses Newton's method with a Jacobian to invert the gradient log-normalizer.
    """
    @jit
    @override
    def to_nat(self) -> NP:
        iterated_function = ExpToNatIteratedFunction[NP, SP](minimum_iterations=1000,
                                                             maximum_iterations=1000,
                                                             rtol=1e-4,
                                                             atol=1e-6,
                                                             z_minimum_iterations=100,
                                                             z_maximum_iterations=1000,
                                                             transform=Adam(1e-1))
        initial_search_parameters = self.initial_search_parameters()
        initial_gt_state = iterated_function.transform.init(initial_search_parameters)
        initial_state = initial_gt_state, initial_search_parameters
        final_state = iterated_function.find_fixed_point(self, initial_state).current_state
        _, final_search_parameters = final_state
        return self.search_to_natural(final_search_parameters)

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
        expectation_parameters: ExpToNat[NP, SP] = natural_parameters.to_exp()
        exp_difference: ExpToNat[NP, SP] = tree_map(jnp.subtract, expectation_parameters, self)
        return type(natural_parameters).unflattened(exp_difference.flattened(),
                                                    **self.fixed_parameters())


@dataclass
class ExpToNatIteratedFunction(
        # The generic parameters are: Parameters, State, Comparand, Differentiand, Trajectory.
        ComparingIteratedFunctionWithCombinator[ExpToNat[NP, SP],
                                                tuple[Any, SP],
                                                SP,
                                                SP,
                                                SP],
        Generic[NP, SP]):
    transform: GradientTransformation[Any, SP] = field()

    @override
    def sampled_state(self, theta: ExpToNat[NP, SP], state: tuple[Any, SP]) -> tuple[Any, SP]:
        current_gt_state, search_parameters = state
        search_gradient = theta.search_gradient(search_parameters)
        transformed_gradient, new_gt_state = self.transform.update(search_gradient,
                                                                   current_gt_state,
                                                                   search_parameters)
        new_search_parameters = tree_map(jnp.add, search_parameters, transformed_gradient)
        return new_gt_state, new_search_parameters

    @override
    def sampled_state_trajectory(
            self,
            theta: ExpToNat[NP, SP],
            augmented: ComparingState[tuple[Any, SP], SP]) -> tuple[tuple[Any, SP], SP]:
        sampled_state = self.sampled_state(theta, augmented.current_state)
        _, trajectory = sampled_state
        return sampled_state, trajectory

    @override
    def extract_comparand(self, state: tuple[Any, SP]) -> SP:
        _, search_parameters = state
        return search_parameters

    @override
    def extract_differentiand(self,
                              theta: ExpToNat[NP, SP],
                              state: tuple[Any, SP]) -> SP:
        _, search_parameters = state
        return search_parameters

    @override
    def implant_differentiand(self,
                              theta: ExpToNat[NP, SP],
                              state: tuple[Any, SP],
                              differentiand: SP) -> tuple[Any, SP]:
        current_gt_state, _ = state
        return current_gt_state, differentiand
