from __future__ import annotations

from tjax import RealArray, dataclass

from .dirichlet import DirichletEP, DirichletNP

__all__ = ['BetaNP', 'BetaEP']


@dataclass
class BetaNP(DirichletNP):
    """
    The Beta distribution.

    The best way to interpret the parameters of the beta distribution are that an observation x in
    [0, 1] represents the Bernoulli probability that outcome 0 (out of {0, 1}) is realized.  In this
    way, the Beta class coincides with a special case of the Dirichlet class.
    """

    # Overridden methods ---------------------------------------------------------------------------
    def to_exp(self) -> BetaEP:
        return BetaEP(super().to_exp().mean_log_probability)

    def sufficient_statistics(self, x: RealArray) -> BetaEP:
        return BetaEP(super().sufficient_statistics(x).mean_log_probability)


@dataclass
class BetaEP(DirichletEP):

    # Overridden methods ---------------------------------------------------------------------------
    def to_nat(self) -> BetaNP:
        return BetaNP(super().to_nat().alpha_minus_one)
