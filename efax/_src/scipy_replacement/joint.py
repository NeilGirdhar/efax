from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np
from numpy.random import Generator
from tjax import NumpyRealArray, NumpyRealNumeric, ShapeLike

from ..tools import join_mappings


@dataclass
class ScipyJointDistribution:
    sub_distributions: dict[str, Any]

    def pdf(self, z: dict[str, Any], out: None = None) -> NumpyRealNumeric:
        joined = join_mappings(sub=self.sub_distributions, z=z)
        return prod(value['sub'].pdf(value['z']) for value in joined.values())

    def rvs(self,
            size: ShapeLike | None = None,
            random_state: Generator | None = None
            ) -> dict[str, Any]:
        return {name: distribution.rvs(size, random_state)
                for name, distribution in self.sub_distributions.items()}

    def entropy(self) -> NumpyRealArray:
        return np.sum([distribution.entropy()
                       for distribution in self.sub_distributions.values()],
                      axis=0)
