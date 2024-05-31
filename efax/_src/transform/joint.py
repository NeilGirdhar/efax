from collections.abc import Callable, Mapping
from functools import reduce
from typing import Any, override

import jax.numpy as jnp
from jax.random import split
from tjax import JaxComplexArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass, field

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parametrization import Distribution, SimpleDistribution
from ..tools import join_mappings


@dataclass
class JointDistribution(Distribution):
    _sub_distributions: Mapping[str, Distribution] = field(
            metadata={'parameter': False})

    @override
    def sub_distributions(self) -> Mapping[str, Distribution]:
        return self._sub_distributions

    def general_method(self,
                       f: Callable[[Distribution], Any],
                       t: type[Distribution] = Distribution
                       ) -> Any:
        return {name: (value.general_method(f, t)
                       if isinstance(value, JointDistribution) else f(value))
                for name, value in self._sub_distributions.items()
                if isinstance(value, JointDistribution | t)}

    def general_sample(self, key: KeyArray, shape: Shape | None = None) -> dict[str, Any]:
        count = 0

        def g(x: Distribution, /) -> None:
            nonlocal count
            count += 1

        self.general_method(g, Samplable)
        keys = split(key, count)
        count = 0

        def f(x: Distribution, /) -> JaxComplexArray:
            assert isinstance(x, Samplable)
            nonlocal keys
            nonlocal count
            retval = x.sample(keys[count], shape)
            count += 1
            return retval

        return self.general_method(f, Samplable)

    def as_dict(self) -> dict[str, Any]:
        def f(x: Distribution, /) -> SimpleDistribution:
            assert isinstance(x, SimpleDistribution)
            return x

        return self.general_method(f)

    @property
    @override
    def shape(self) -> Shape:
        for distribution in self._sub_distributions.values():
            return distribution.shape
        raise ValueError


@dataclass
class JointDistributionE(JointDistribution,
                         HasEntropyEP['JointDistributionN'],
                         ExpectationParametrization['JointDistributionN']):
    _sub_distributions: Mapping[str, ExpectationParametrization[Any]] = field(
            metadata={'parameter': False})

    @override
    def sub_distributions(self) -> Mapping[str, ExpectationParametrization[Any]]:
        return self._sub_distributions

    @override
    @classmethod
    def natural_parametrization_cls(cls) -> type['JointDistributionN']:
        return JointDistributionN

    @override
    def to_nat(self) -> 'JointDistributionN':
        return JointDistributionN({name: value.to_nat()
                                   for name, value in self._sub_distributions.items()})

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        def f(x: ExpectationParametrization[Any]) -> JaxRealArray:
            assert isinstance(x, HasEntropyEP)
            return x.expected_carrier_measure()

        return reduce(jnp.add, (f(value) for value in self._sub_distributions.values()))


@dataclass
class JointDistributionN(JointDistribution,
                         HasEntropyNP[JointDistributionE],
                         NaturalParametrization[JointDistributionE, dict[str, Any]]):
    _sub_distributions: Mapping[str, NaturalParametrization[Any, Any]] = field(
            metadata={'parameter': False})

    @override
    def sub_distributions(self) -> Mapping[str, NaturalParametrization[Any, Any]]:
        return self._sub_distributions

    @override
    def log_normalizer(self) -> JaxRealArray:
        return reduce(jnp.add,
                      (x.log_normalizer() for x in self._sub_distributions.values()))

    @override
    def to_exp(self) -> JointDistributionE:
        return JointDistributionE({name: value.to_exp()
                                   for name, value in self._sub_distributions.items()})

    @override
    def carrier_measure(self, x: dict[str, Any]) -> JaxRealArray:
        joined = join_mappings(sub=self._sub_distributions, x=x)
        return reduce(jnp.add,
                      (value['sub'].carrier_measure(value['x'])
                       for value in joined.values()))

    @override
    @classmethod
    def sufficient_statistics(cls, x: dict[str, Any], **fixed_parameters: Any
                              ) -> JointDistributionE:
        raise NotImplementedError
