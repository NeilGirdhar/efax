from __future__ import annotations

from abc import abstractmethod

from tjax import JaxArray, KeyArray, Shape

from efax._src.parametrization import SimpleDistribution


class Samplable(SimpleDistribution):
    """A SimpleDistribution that can draw samples using a JAX PRNG key."""

    @abstractmethod
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxArray:
        """Draw samples from this distribution.

        Args:
            key: A JAX PRNG key.
            shape: The batch shape of samples to draw.  If None, returns a single sample
                with the same shape as this distribution.
        """
        raise NotImplementedError
