from __future__ import annotations

from abc import abstractmethod

from efax._src.parametrization import SimpleDistribution


class Multidimensional(SimpleDistribution):
    """A SimpleDistribution whose domain is a vector (or matrix) rather than a scalar.

    The dimensionality is a static property of the distribution — it determines how many
    elements each sample occupies and how support constraints are applied per element.
    """

    @abstractmethod
    def dimensions(self) -> int:
        """Return the number of elements in each sample from this distribution."""
        raise NotImplementedError
