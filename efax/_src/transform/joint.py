from collections.abc import Mapping
from functools import reduce
from typing import Any, override

import jax.numpy as jnp
from tjax import JaxRealArray, Shape
from tjax.dataclasses import dataclass, field

from ..expectation_parametrization import ExpectationParametrization
from ..natural_parametrization import NaturalParametrization
from ..parametrization import Parametrization


@dataclass
class JointDistribution(Parametrization):
    sub_distributions_objects: Mapping[str, Parametrization] = field(metadata={'parameter': False})

    @property
    @override
    def shape(self) -> Shape:
        first = next(iter(self.sub_distributions_objects.values()))
        return first.shape

    @override
    def sub_distributions(self) -> Mapping[str, Parametrization]:
        return self.sub_distributions_objects


@dataclass
class JointDistributionE(JointDistribution,
                         ExpectationParametrization['JointDistributionN']):
    sub_distributions_objects: Mapping[str, ExpectationParametrization[Any]] = field(
            metadata={'parameter': False})

    @override
    def sub_distributions(self) -> Mapping[str, ExpectationParametrization[Any]]:
        return self.sub_distributions_objects

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type['JointDistributionN']:
        return JointDistributionN

    @override
    def to_nat(self) -> 'JointDistributionN':
        return JointDistributionN({name: value.to_nat()
                                   for name, value in self.sub_distributions_objects.items()})


@dataclass
class JointDistributionN(JointDistribution,
                         NaturalParametrization[JointDistributionE, dict[str, Any]]):
    sub_distributions_objects: Mapping[str, NaturalParametrization[Any, Any]] = field(
            metadata={'parameter': False})

    @override
    def sub_distributions(self) -> Mapping[str, NaturalParametrization[Any, Any]]:
        return self.sub_distributions_objects

    @override
    def log_normalizer(self) -> JaxRealArray:
        return reduce(jnp.add,
                      (x.log_normalizer() for x in self.sub_distributions_objects.values()))

    @override
    def to_exp(self) -> JointDistributionE:
        return JointDistributionE({name: value.to_exp()
                                   for name, value in self.sub_distributions_objects.items()})

    @override
    def carrier_measure(self, x: dict[str, Any]) -> JaxRealArray:
        return reduce(jnp.add,
                      (value.carrier_measure(x[name])
                       for name, value in self.sub_distributions_objects.items()))

    @classmethod
    @override
    def sufficient_statistics(cls, x: dict[str, Any], **fixed_parameters: Any
                              ) -> JointDistributionE:
        sub_distributions_classes = fixed_parameters.pop('sub_distributions_classes')
        return JointDistributionE(
            {name: value.sufficient_statistics(x[name], **fixed_parameters.get(name, {}))
             for name, value in sub_distributions_classes.items()})
