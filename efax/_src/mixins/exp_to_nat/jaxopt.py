from typing import Any

from jax import jit
from jaxopt import Broyden
from tjax import JaxRealArray
from tjax.dataclasses import dataclass
from typing_extensions import override

from .exp_to_nat import ExpToNat, ExpToNatMinimizer


@dataclass
class IteratedMinimizer(ExpToNatMinimizer):
    broyden_parms: dict[str, Any]

    @override
    def solve(self, exp_to_nat: ExpToNat[Any]) -> JaxRealArray:
        @jit
        def f(x: JaxRealArray, /) -> JaxRealArray:
            return exp_to_nat.search_gradient(x)

        solver = Broyden(f, **self.broyden_parms)
        results = solver.run(exp_to_nat.initial_search_parameters())
        return results.params


default_minimizer = IteratedMinimizer({'maxiter': 2000,
                                       'tol': 1e-5})
