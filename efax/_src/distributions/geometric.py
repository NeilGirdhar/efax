from __future__ import annotations

import jax.random as jr
from tjax import JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import ScalarSupport, distribution_parameter, negative_support, positive_support
from ..parametrization import SimpleDistribution
from .negative_binomial_common import NBCommonEP, NBCommonNP


@dataclass
class GeometricNP(HasEntropyNP['GeometricEP'],
                  Samplable,
                  NBCommonNP['GeometricEP'],
                  NaturalParametrization['GeometricEP', JaxRealArray],
                  SimpleDistribution):
    """The natural parameters of the geometric distribution.

    Models the number of Bernoulli trials having probability p until one failure.  Thus, it has
    support {0, ...}.

    Args:
        log_not_p: log(1-p).
    """
    log_not_p: JaxRealArray = distribution_parameter(ScalarSupport(ring=negative_support))

    @override
    def to_exp(self) -> GeometricEP:
        return GeometricEP(self._mean())

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray
                              ) -> GeometricEP:
        return GeometricEP(x)

    @override
    def _failures(self) -> int:
        return 1

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_exp().sample(key, shape)


@dataclass
class GeometricEP(HasEntropyEP[GeometricNP],
                  Samplable,
                  NBCommonEP[GeometricNP],
                  SimpleDistribution):
    """The expectation parameters of the geometric distribution.

    Models the number of Bernoulli trials having probability p until one failure.  Thus, it has
    support {0, ...}.

    Args:
        mean: E(x).
    """
    mean: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[GeometricNP]:
        return GeometricNP

    @override
    def to_nat(self) -> GeometricNP:
        return GeometricNP(self._log_not_p())

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.zeros(self.mean.shape)

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        xp = self.array_namespace()
        p = xp.reciprocal(self.mean)
        return jr.geometric(key, p, shape)

    @override
    def _failures(self) -> int:
        return 1
