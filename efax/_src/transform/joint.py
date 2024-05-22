from collections.abc import Callable, Mapping
from functools import reduce
from typing import Any, TypeVar, override

import jax.numpy as jnp
from tjax import JaxComplexArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass, field

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parametrization import GeneralParametrization, Parametrization
from ..tools import join_mappings

T = TypeVar('T', bound=GeneralParametrization)


@dataclass
class JointDistribution(GeneralParametrization):
    # TODO: rename _sub_distributions
    sub_distributions_objects: Mapping[str, GeneralParametrization] = field(
            metadata={'parameter': False})

    @override
    def sub_distributions(self) -> Mapping[str, GeneralParametrization]:
        return self.sub_distributions_objects

    def general_method(self,
                       f: Callable[[T], Any],
                       t: type[T] = GeneralParametrization
                       ) -> Any:
        return {name: (value.general_method(f, t)
                       if isinstance(value, JointDistribution) else f(value))
                for name, value in self.sub_distributions_objects.items()
                if isinstance(value, JointDistribution | t)}

    def general_sample(self, key: KeyArray, shape: Shape | None = None) -> dict[str, Any]:
        def f(x: GeneralParametrization, /) -> JaxComplexArray:
            assert isinstance(x, Samplable)
            return x.sample(key, shape)

        return self.general_method(f, Samplable)

    def as_dict(self) -> dict[str, Any]:
        def f(x: GeneralParametrization, /) -> Parametrization:
            assert isinstance(x, Parametrization)
            return x

        return self.general_method(f)

    @property
    @override
    def shape(self) -> Shape:
        for distribution in self.sub_distributions_objects.values():
            return distribution.shape
        raise ValueError


@dataclass
class JointDistributionE(JointDistribution,
                         HasEntropyEP['JointDistributionN'],
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

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        def f(x: ExpectationParametrization[Any]) -> JaxRealArray:
            assert isinstance(x, HasEntropyEP)
            return x.expected_carrier_measure()

        return reduce(jnp.add, (f(value) for value in self.sub_distributions_objects.values()))


@dataclass
class JointDistributionN(JointDistribution,
                         HasEntropyNP[JointDistributionE],
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
        joined = join_mappings(sub=self.sub_distributions_objects, x=x)
        return reduce(jnp.add,
                      (value['sub'].carrier_measure(value['x'])
                       for value in joined.values()))

    @classmethod
    @override
    def sufficient_statistics(cls, x: dict[str, Any], **fixed_parameters: Any
                              ) -> JointDistributionE:
        sub_distributions_classes = fixed_parameters.pop('sub_distributions_classes')
        joined = join_mappings(sub=sub_distributions_classes, fixed=fixed_parameters, x=x)
        return JointDistributionE(
                {name: value['sub'].sufficient_statistics(value['x'], **value.get('fixed', {}))
                 for name, value in joined.items()})
