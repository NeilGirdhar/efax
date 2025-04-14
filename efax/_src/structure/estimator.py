from dataclasses import fields
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

from array_api_compat import array_namespace
from tjax import JaxArray, JaxComplexArray, JaxRealArray
from tjax.dataclasses import dataclass

from ..iteration import flatten_mapping, parameters
from ..parametrization import Distribution, SimpleDistribution
from ..types import Path
from .parameter_names import parameter_names
from .structure import Structure, SubDistributionInfo

if TYPE_CHECKING:
    from ..natural_parametrization import NaturalParametrization

T = TypeVar('T')
P = TypeVar('P', bound=Distribution)
SP = TypeVar('SP', bound=SimpleDistribution)


@dataclass
class MaximumLikelihoodEstimator(Structure[P]):
    """This class does maximum likelihood estimation.

    To do this, it needs to store the structure and the fixed parameters.
    """
    fixed_parameters: dict[Path, JaxComplexArray]

    @classmethod
    def create_simple_estimator(cls,
                                type_p: type[SimpleDistribution],
                                **fixed_parameters: JaxArray
                                ) -> 'MaximumLikelihoodEstimator[Any]':
        """Create an estimator for a simple expectation parametrization class.

        This doesn't work with things like JointDistributionE.
        """
        from ..expectation_parametrization import ExpectationParametrization  # noqa: PLC0415
        assert issubclass(type_p, ExpectationParametrization)
        return MaximumLikelihoodEstimator(
                [SubDistributionInfo((), type_p, 0, [])],
                {(name,): value for name, value in fixed_parameters.items()})

    @classmethod
    def create_estimator(cls, p: P) -> Self:
        """Create an estimator for an expectation parametrization."""
        from ..expectation_parametrization import ExpectationParametrization  # noqa: PLC0415
        infos = cls.create(p).infos
        assert isinstance(p, ExpectationParametrization)
        fixed_parameters = parameters(p, fixed=True)
        return cls(infos, fixed_parameters)

    @classmethod
    def create_estimator_from_natural(cls, p: 'NaturalParametrization[Any, Any]'
                                      ) -> 'MaximumLikelihoodEstimator[Any]':
        """Create an estimator for a natural parametrization."""
        infos = MaximumLikelihoodEstimator.create(p).to_exp().infos
        fixed_parameters = parameters(p, fixed=True)
        return cls(infos, fixed_parameters)

    def sufficient_statistics(self, x: dict[str, Any] | JaxComplexArray) -> P:
        from ..expectation_parametrization import ExpectationParametrization  # noqa: PLC0415
        from ..natural_parametrization import NaturalParametrization  # noqa: PLC0415
        from ..transform.joint import JointDistributionE  # noqa: PLC0415
        constructed: dict[Path, ExpectationParametrization[Any]] = {}

        def g(info: SubDistributionInfo, x: JaxComplexArray) -> None:
            assert not info.sub_distribution_names
            exp_cls = info.type_
            assert issubclass(exp_cls, ExpectationParametrization)

            nat_cls = exp_cls.natural_parametrization_cls()
            assert issubclass(nat_cls, NaturalParametrization)

            fixed_parameters: dict[str, Any] = {name: self.fixed_parameters[*info.path, name]
                                                for name in parameter_names(nat_cls, fixed=True)}
            p = nat_cls.sufficient_statistics(x, **fixed_parameters)
            assert isinstance(p, ExpectationParametrization)
            constructed[info.path] = p

        def h(info: SubDistributionInfo) -> None:
            exp_cls = info.type_
            assert issubclass(exp_cls, JointDistributionE)
            sub_distributions = {name: constructed[*info.path, name]
                                 for name in info.sub_distribution_names}
            constructed[info.path] = exp_cls(sub_distributions)

        if isinstance(x, dict):
            flat_x = flatten_mapping(x)
            for info in self.infos:
                if info.path in flat_x:
                    g(info, flat_x[info.path])
                else:
                    h(info)
        else:
            info, = self.infos
            g(info, x)
        return cast('P', constructed[()])

    def from_conjugate_prior_distribution(self,
                                          cp: 'NaturalParametrization[Any, Any]'
                                          ) -> tuple[P, JaxRealArray]:
        from ..interfaces.conjugate_prior import HasConjugatePrior  # noqa: PLC0415
        from ..transform.joint import JointDistributionN  # noqa: PLC0415
        constructed: dict[Path, Distribution] = {}
        n = None
        for info in self.infos:
            assert issubclass(info.type_, HasConjugatePrior)
            fixed_parameters = {
                    this_field.name: self.fixed_parameters[*info.path, this_field.name]
                    for this_field in fields(info.type_)
                    if this_field.metadata.get('parameter', False) and this_field.metadata['fixed']}
            cp_i = cp
            for path_element in info.path:
                assert isinstance(cp_i, JointDistributionN)
                cp_i = cp_i.sub_distributions()[path_element]
            p, n_i = info.type_.from_conjugate_prior_distribution(cp, **fixed_parameters)
            if n is None:
                n = n_i
            else:
                xp = array_namespace(n)
                assert xp.all(n == n_i)
            constructed[info.path] = p
        assert n is not None
        return_p = constructed[()]
        return cast('P', return_p), n
