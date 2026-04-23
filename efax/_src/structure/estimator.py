from dataclasses import fields
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

from array_api_compat import array_namespace
from tjax import JaxArray, JaxComplexArray, JaxRealArray
from tjax.dataclasses import dataclass

from efax._src.iteration import flatten_mapping, parameters
from efax._src.parametrization import Distribution, SimpleDistribution
from efax._src.types import Path

from .assembler import Assembler, JointDistributionInfo, SimpleDistributionInfo
from .parameter_names import parameter_names

if TYPE_CHECKING:
    from efax._src.natural_parametrization import NaturalParametrization

    NP = TypeVar("NP", bound=NaturalParametrization)

P = TypeVar("P", bound=Distribution)
SP = TypeVar("SP", bound=SimpleDistribution)


@dataclass
class Estimator(Assembler[P]):
    """An Assembler that also performs maximum likelihood estimation.

    Extends Assembler by tracking which parameters are fixed (held constant during estimation)
    and providing operations to estimate the free parameters from observed data.

    In exponential family distributions, maximum likelihood estimation reduces to computing
    sufficient statistics, making sufficient_statistics the core estimation operation.
    """

    fixed_parameters: dict[Path, JaxComplexArray]

    @classmethod
    def from_type(cls, type_p: type[SP], **fixed_parameters: JaxArray) -> "Estimator[SP]":
        """Create an Estimator from a simple ExpectationParametrization class.

        Use this when you have a type rather than an instance.  Does not work with composite
        distributions such as JointDistributionE.
        """
        from efax._src.expectation_parametrization import (  # noqa: PLC0415
            ExpectationParametrization,
        )

        if not issubclass(type_p, ExpectationParametrization):
            msg = f"{type_p.__name__} is not an EP"
            raise TypeError(msg)
        return Estimator(
            [SimpleDistributionInfo((), type_p, 0)],
            {(name,): value for name, value in fixed_parameters.items()},
        )

    @classmethod
    def from_expectation(cls, p: P) -> Self:
        """Create an Estimator from an expectation-parametrized distribution.

        Extracts the distribution tree structure from p and records which of its parameters
        are marked fixed, so they will be held constant during estimation.
        """
        from efax._src.expectation_parametrization import (  # noqa: PLC0415
            ExpectationParametrization,
        )

        infos = cls.create_assembler(p).infos
        if not isinstance(p, ExpectationParametrization):
            msg = f"{type(p).__name__} is not an EP"
            raise TypeError(msg)
        fixed_parameters = parameters(p, fixed=True)
        return cls(infos, fixed_parameters)

    @classmethod
    def from_natural(cls, p: "NP") -> "Estimator[NP]":
        """Create an Estimator from a natural-parametrized distribution.

        Converts the distribution tree to expectation parametrization types while preserving
        the fixed parameter values from p.
        """
        infos = Estimator.create_assembler(p).to_exp().infos  # type: ignore
        fixed_parameters = parameters(p, fixed=True)
        return Estimator(infos, fixed_parameters)

    def sufficient_statistics(self, x: dict[str, Any] | JaxComplexArray) -> P:
        """Compute the sufficient statistics of observation x.

        In exponential family distributions, sufficient statistics are exactly the MLE parameter
        estimates, so this is the primary estimation operation.  Fixed parameters are supplied
        automatically from this Estimator's stored values.
        """
        from efax._src.expectation_parametrization import (  # noqa: PLC0415
            ExpectationParametrization,
        )
        from efax._src.natural_parametrization import NaturalParametrization  # noqa: PLC0415
        from efax._src.transform.joint import JointDistributionE  # noqa: PLC0415

        constructed: dict[Path, ExpectationParametrization] = {}

        def g(info: SimpleDistributionInfo, x: JaxComplexArray) -> None:
            exp_cls = info.type_
            assert issubclass(exp_cls, ExpectationParametrization)

            nat_cls = exp_cls.natural_parametrization_cls()
            assert issubclass(nat_cls, NaturalParametrization)

            fixed_parameters: dict[str, Any] = {
                name: self.fixed_parameters[*info.path, name]
                for name in parameter_names(nat_cls, fixed=True)
            }
            p = nat_cls.sufficient_statistics(x, **fixed_parameters)  # type: ignore
            assert isinstance(p, ExpectationParametrization)
            constructed[info.path] = p

        def h(info: JointDistributionInfo) -> None:
            exp_cls = info.type_
            assert issubclass(exp_cls, JointDistributionE)
            sub_distributions = {
                name: constructed[*info.path, name] for name in info.sub_distribution_names
            }
            constructed[info.path] = exp_cls(sub_distributions)

        if isinstance(x, JaxArray):
            (info,) = self.infos
            assert isinstance(info, SimpleDistributionInfo)
            g(info, x)
        else:
            flat_x = flatten_mapping(x)
            for info in self.infos:
                if info.path in flat_x:
                    assert isinstance(info, SimpleDistributionInfo)
                    g(info, flat_x[info.path])
                else:
                    assert isinstance(info, JointDistributionInfo)
                    h(info)
        return cast("P", constructed[()])

    def from_conjugate_prior(self, cp: "NaturalParametrization") -> tuple[P, JaxRealArray]:
        """Recover distribution parameters and observation count from a conjugate prior.

        Given a conjugate prior distribution cp, returns the distribution p whose parameters
        are encoded in cp together with the pseudo-observation count n.  Fixed parameters are
        supplied automatically from this Estimator's stored values.
        """
        from efax._src.interfaces.conjugate_prior import HasConjugatePrior  # noqa: PLC0415
        from efax._src.transform.joint import JointDistributionN  # noqa: PLC0415

        constructed: dict[Path, Distribution] = {}
        n = None
        for info in self.infos:
            if isinstance(info, JointDistributionInfo):
                sub_distributions = {
                    name: constructed[*info.path, name] for name in info.sub_distribution_names
                }
                constructed[info.path] = info.type_(_sub_distributions=sub_distributions)
                continue
            assert issubclass(info.type_, HasConjugatePrior)
            fixed_parameters = {
                this_field.name: self.fixed_parameters[*info.path, this_field.name]
                for this_field in fields(info.type_)
                if this_field.metadata.get("parameter", False) and this_field.metadata["fixed"]
            }
            cp_i = cp
            for path_element in info.path:
                assert isinstance(cp_i, JointDistributionN)
                cp_i = cp_i.sub_distributions()[path_element]
            p, n_i = info.type_.from_conjugate_prior_distribution(cp_i, **fixed_parameters)
            if n is None:
                n = n_i
            else:
                xp = array_namespace(n)
                assert xp.all(n == n_i)
            constructed[info.path] = p
        assert n is not None
        return_p = constructed[()]
        return cast("P", return_p), n
