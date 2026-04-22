from collections.abc import Callable, Mapping
from functools import reduce
from typing import Any, override

from array_api_compat import array_namespace
from tjax import JaxArray, JaxComplexArray, JaxRealArray, KeyArray, RngStream, Shape
from tjax.dataclasses import dataclass

from efax._src.expectation_parametrization import ExpectationParametrization
from efax._src.interfaces.samplable import Samplable
from efax._src.mixins.has_entropy import HasEntropyEP, HasEntropyNP
from efax._src.natural_parametrization import NaturalParametrization
from efax._src.parametrization import Distribution, SimpleDistribution
from efax._src.tools import join_mappings


@dataclass
class JointDistribution(Distribution):
    _sub_distributions: Mapping[str, Distribution]

    @override
    def sub_distributions(self) -> Mapping[str, Distribution]:
        return self._sub_distributions

    def general_method(
        self, f: Callable[[Distribution], Any], t: type[Distribution] = Distribution
    ) -> dict[str, Any]:
        return {
            name: (value.general_method(f, t) if isinstance(value, JointDistribution) else f(value))
            for name, value in self._sub_distributions.items()
            if isinstance(value, (JointDistribution, t))
        }

    def general_sample(self, key: KeyArray, shape: Shape | None = None) -> dict[str, Any]:
        stream = RngStream(key)

        def f(x: Distribution, /) -> JaxComplexArray:
            assert isinstance(x, Samplable)
            return x.sample(stream.key(), shape)

        return self.general_method(f, Samplable)

    def as_dict(self) -> dict[str, Any]:
        def f(x: Distribution, /) -> SimpleDistribution:
            assert isinstance(x, SimpleDistribution)
            return x

        return self.general_method(f)

    @property
    @override
    def shape(self) -> Shape:
        shapes = [d.shape for d in self._sub_distributions.values()]
        if not shapes:
            msg = "JointDistribution has no sub-distributions"
            raise ValueError(msg)
        shape = shapes[0]
        if any(s != shape for s in shapes[1:]):
            names = list(self._sub_distributions)
            msg = (
                f"Sub-distribution shapes are inconsistent: {dict(zip(names, shapes, strict=True))}"
            )
            raise ValueError(msg)
        return shape


@dataclass
class JointDistributionE(JointDistribution, HasEntropyEP["JointDistributionN"]):
    _sub_distributions: Mapping[str, ExpectationParametrization]

    @override
    def sub_distributions(self) -> Mapping[str, ExpectationParametrization]:
        return self._sub_distributions

    @override
    @classmethod
    def natural_parametrization_cls(cls) -> type["JointDistributionN"]:
        return JointDistributionN

    @override
    def to_nat(self) -> "JointDistributionN":
        return JointDistributionN(
            {name: value.to_nat() for name, value in self._sub_distributions.items()}
        )

    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = array_namespace(self)

        def f(x: ExpectationParametrization) -> JaxRealArray:
            assert isinstance(x, HasEntropyEP)
            return x.expected_carrier_measure()

        return reduce(xp.add, (f(value) for value in self._sub_distributions.values()))


@dataclass
class JointDistributionN(
    JointDistribution,
    HasEntropyNP[JointDistributionE],
    NaturalParametrization[JointDistributionE, dict[str, Any]],
):
    _sub_distributions: Mapping[str, NaturalParametrization]

    @override
    def sub_distributions(self) -> Mapping[str, NaturalParametrization]:
        return self._sub_distributions

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = array_namespace(self)

        return reduce(xp.add, (x.log_normalizer() for x in self._sub_distributions.values()))

    @override
    def to_exp(self) -> JointDistributionE:
        return JointDistributionE(
            {name: value.to_exp() for name, value in self._sub_distributions.items()}
        )

    @override
    def carrier_measure(self, x: dict[str, Any]) -> JaxRealArray:
        xp = array_namespace(self)

        joined = join_mappings(sub=self._sub_distributions, x=x)
        return reduce(
            xp.add, (value["sub"].carrier_measure(value["x"]) for value in joined.values())
        )

    @override
    @classmethod
    def sufficient_statistics(
        cls, x: dict[str, Any], **fixed_parameters: JaxArray
    ) -> JointDistributionE:
        raise NotImplementedError
