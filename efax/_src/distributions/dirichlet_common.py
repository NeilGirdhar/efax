from __future__ import annotations

from typing import Any, Generic, TypeVar

import jax.random as jr
import jax.scipy.special as jss
from tjax import JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override

from ..interfaces.multidimensional import Multidimensional
from ..interfaces.samplable import Samplable
from ..mixins.exp_to_nat.exp_to_nat import ExpToNat
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import RealField, VectorSupport, distribution_parameter, negative_support

EP = TypeVar('EP', bound='DirichletCommonEP[Any]')


@dataclass
class DirichletCommonNP(HasEntropyNP[EP],
                        Samplable,
                        Multidimensional,
                        NaturalParametrization[EP, JaxRealArray],
                        Generic[EP]):
    alpha_minus_one: JaxRealArray = distribution_parameter(VectorSupport(
        ring=RealField(minimum=-1.0, generation_scale=3.0)))

    @property
    @override
    def shape(self) -> Shape:
        return self.alpha_minus_one.shape[:-1]

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.array_namespace()
        q = self.alpha_minus_one
        return (xp.sum(jss.gammaln(q + 1.0), axis=-1)
                - jss.gammaln(xp.sum(q, axis=-1) + self.dimensions()))

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        if shape is not None:
            shape += self.shape
        return jr.dirichlet(key, 1.0 + self.alpha_minus_one, shape)[..., :-1]

    @override
    def dimensions(self) -> int:
        return self.alpha_minus_one.shape[-1]

    def _exp_helper(self) -> JaxRealArray:
        xp = self.array_namespace()
        q = self.alpha_minus_one
        return jss.digamma(q + 1.0) - jss.digamma(xp.sum(q, axis=-1, keepdims=True) + q.shape[-1])


NP = TypeVar('NP', bound=DirichletCommonNP[Any])


@dataclass
class DirichletCommonEP(HasEntropyEP[NP],
                        Samplable,
                        ExpToNat[NP],
                        Multidimensional, Generic[NP]):
    mean_log_probability: JaxRealArray = distribution_parameter(VectorSupport(
        ring=negative_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.mean_log_probability.shape[:-1]

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.zeros(self.shape)

    @override
    def dimensions(self) -> int:
        return self.mean_log_probability.shape[-1]

    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_nat().sample(key, shape)
