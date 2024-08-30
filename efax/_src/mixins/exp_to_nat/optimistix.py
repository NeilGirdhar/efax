from typing import Any

import jax.numpy as jnp
import optimistix as optx
from jax import jit
from tjax import JaxRealArray
from tjax.dataclasses import dataclass, field
from tjax.gradient import Adam
from typing_extensions import override

from .exp_to_nat import ExpToNat, ExpToNatMinimizer

type RootFinder[Y, Out, Aux] = (
        optx.AbstractRootFinder[Y, Out, Aux, Any]
        | optx.AbstractLeastSquaresSolver[Y, Out, Aux, Any]
        | optx.AbstractMinimiser[Y, Aux, Any])


@dataclass
class OptimistixRootFinder(ExpToNatMinimizer):
    solver: RootFinder[JaxRealArray, JaxRealArray, Any] = field(static=True)
    max_steps: int = field(static=True)
    send_lower_and_upper: bool = field(static=True, default=False)

    @override
    def solve(self, exp_to_nat: ExpToNat[Any]) -> JaxRealArray:
        @jit
        def f(x: JaxRealArray, args: None, /) -> JaxRealArray:
            if self.send_lower_and_upper:
                x = x[jnp.newaxis]
            retval = exp_to_nat.search_gradient(x)
            if self.send_lower_and_upper:
                retval = retval[0]
            return retval

        initial = exp_to_nat.initial_search_parameters()
        if self.send_lower_and_upper:
            options = {}
            assert initial.shape == (1,)
            initial = initial[0]
            options['lower'] = initial - 1.0
            options['upper'] = initial + 1.0
        else:
            options = None
        results: optx.Solution[JaxRealArray, None] = optx.root_find(
                f, self.solver, initial, max_steps=self.max_steps, throw=False, options=options)
        if self.send_lower_and_upper:
            return results.value[jnp.newaxis]
        return results.value


default_minimizer = OptimistixRootFinder(
        solver=optx.OptaxMinimiser[JaxRealArray, None](optim=Adam(5e-2), rtol=0.0, atol=1e-7),
        max_steps=1000)


default_bisection_minimizer = OptimistixRootFinder(
        solver=optx.Bisection(  # type: ignore[call-arg]
            rtol=0.0, atol=1e-7, flip='detect', expand_if_necessary=True),
        max_steps=1000,
        send_lower_and_upper=True)
