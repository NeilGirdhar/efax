"""The EFAX Library."""

from ._src.distributions.bernoulli import BernoulliEP, BernoulliNP
from ._src.distributions.beta import BetaEP, BetaNP
from ._src.distributions.categorical import CategoricalEP, CategoricalNP
from ._src.distributions.chi import ChiEP, ChiNP
from ._src.distributions.chi_square import ChiSquareEP, ChiSquareNP
from ._src.distributions.cmvn.circularly_symmetric import (
    ComplexCircularlySymmetricNormalEP,
    ComplexCircularlySymmetricNormalNP,
)
from ._src.distributions.cmvn.unit_variance import (
    ComplexMultivariateUnitVarianceNormalEP,
    ComplexMultivariateUnitVarianceNormalNP,
)
from ._src.distributions.complex_normal.complex_normal import ComplexNormalEP, ComplexNormalNP
from ._src.distributions.complex_normal.unit_variance import (
    ComplexUnitVarianceNormalEP,
    ComplexUnitVarianceNormalNP,
)
from ._src.distributions.complex_von_mises import ComplexVonMisesEP, ComplexVonMisesNP
from ._src.distributions.dirichlet import DirichletEP, DirichletNP
from ._src.distributions.exponential import ExponentialEP, ExponentialNP
from ._src.distributions.gamma import GammaEP, GammaNP, GammaVP
from ._src.distributions.gen_dirichlet import GeneralizedDirichletEP, GeneralizedDirichletNP
from ._src.distributions.geometric import GeometricEP, GeometricNP
from ._src.distributions.inverse_gamma import InverseGammaEP, InverseGammaNP
from ._src.distributions.inverse_gaussian import InverseGaussianEP, InverseGaussianNP
from ._src.distributions.log_normal.log_normal import LogNormalEP, LogNormalNP
from ._src.distributions.log_normal.unit_variance import (
    UnitVarianceLogNormalEP,
    UnitVarianceLogNormalNP,
)
from ._src.distributions.logarithmic import LogarithmicEP, LogarithmicNP
from ._src.distributions.multivariate_normal.arbitrary import (
    MultivariateNormalEP,
    MultivariateNormalNP,
    MultivariateNormalVP,
)
from ._src.distributions.multivariate_normal.diagonal import (
    MultivariateDiagonalNormalEP,
    MultivariateDiagonalNormalNP,
    MultivariateDiagonalNormalVP,
)
from ._src.distributions.multivariate_normal.fixed_variance import (
    MultivariateFixedVarianceNormalEP,
    MultivariateFixedVarianceNormalNP,
)
from ._src.distributions.multivariate_normal.isotropic import IsotropicNormalEP, IsotropicNormalNP
from ._src.distributions.multivariate_normal.unit_variance import (
    MultivariateUnitVarianceNormalEP,
    MultivariateUnitVarianceNormalNP,
)
from ._src.distributions.negative_binomial import NegativeBinomialEP, NegativeBinomialNP
from ._src.distributions.normal.normal import NormalDP, NormalEP, NormalNP, NormalVP
from ._src.distributions.normal.unit_variance import UnitVarianceNormalEP, UnitVarianceNormalNP
from ._src.distributions.poisson import PoissonEP, PoissonNP
from ._src.distributions.rayleigh import RayleighEP, RayleighNP
from ._src.distributions.softplus_normal.softplus import SoftplusNormalEP, SoftplusNormalNP
from ._src.distributions.softplus_normal.unit_variance import (
    UnitVarianceSoftplusNormalEP,
    UnitVarianceSoftplusNormalNP,
)
from ._src.distributions.von_mises import VonMisesFisherEP, VonMisesFisherNP
from ._src.distributions.weibull import WeibullEP, WeibullNP
from ._src.distributions.wishart import WishartEP, WishartNP
from ._src.expectation_parametrization import (
    ExpectationParametrization,
    expectation_parameters_from_characteristic_function,
)
from ._src.interfaces.conjugate_prior import HasConjugatePrior, HasGeneralizedConjugatePrior
from ._src.interfaces.multidimensional import Multidimensional
from ._src.interfaces.samplable import Samplable
from ._src.iteration import (
    flat_dict_of_observations,
    flat_dict_of_parameters,
    flatten_mapping,
    parameters,
    unflatten_mapping,
)
from ._src.mixins.has_entropy import HasEntropy, HasEntropyEP, HasEntropyNP
from ._src.natural_parametrization import NaturalParametrization
from ._src.parameter.parameter import distribution_parameter
from ._src.parameter.ring import BooleanRing, ComplexField, IntegralRing, RealField, Ring
from ._src.parameter.support import (
    CircularBoundedSupport,
    ScalarSupport,
    SimplexSupport,
    SquareMatrixSupport,
    SubsimplexSupport,
    Support,
    SymmetricMatrixSupport,
    VectorSupport,
)
from ._src.parametrization import Distribution, SimpleDistribution
from ._src.structure.assembler import (
    Assembler,
    JointDistributionInfo,
    SimpleDistributionInfo,
    SubDistributionInfo,
)
from ._src.structure.estimator import Estimator
from ._src.structure.flattener import Flattener
from ._src.tools import (
    parameter_dot_product,
    parameter_holomorphic_dot,
    parameter_map,
    parameter_mean,
)
from ._src.transform.joint import JointDistribution, JointDistributionE, JointDistributionN

__all__ = [
    "Assembler",
    "BernoulliEP",
    "BernoulliNP",
    "BetaEP",
    "BetaNP",
    "BooleanRing",
    "CategoricalEP",
    "CategoricalNP",
    "ChiEP",
    "ChiNP",
    "ChiSquareEP",
    "ChiSquareNP",
    "CircularBoundedSupport",
    "ComplexCircularlySymmetricNormalEP",
    "ComplexCircularlySymmetricNormalNP",
    "ComplexField",
    "ComplexMultivariateUnitVarianceNormalEP",
    "ComplexMultivariateUnitVarianceNormalNP",
    "ComplexNormalEP",
    "ComplexNormalNP",
    "ComplexUnitVarianceNormalEP",
    "ComplexUnitVarianceNormalNP",
    "ComplexVonMisesEP",
    "ComplexVonMisesNP",
    "DirichletEP",
    "DirichletNP",
    "Distribution",
    "Estimator",
    "ExpectationParametrization",
    "ExponentialEP",
    "ExponentialNP",
    "Flattener",
    "GammaEP",
    "GammaNP",
    "GammaVP",
    "GeneralizedDirichletEP",
    "GeneralizedDirichletNP",
    "GeometricEP",
    "GeometricNP",
    "HasConjugatePrior",
    "HasEntropy",
    "HasEntropyEP",
    "HasEntropyNP",
    "HasGeneralizedConjugatePrior",
    "IntegralRing",
    "InverseGammaEP",
    "InverseGammaNP",
    "InverseGaussianEP",
    "InverseGaussianNP",
    "IsotropicNormalEP",
    "IsotropicNormalNP",
    "JointDistribution",
    "JointDistributionE",
    "JointDistributionInfo",
    "JointDistributionN",
    "LogNormalEP",
    "LogNormalNP",
    "LogarithmicEP",
    "LogarithmicNP",
    "Multidimensional",
    "MultivariateDiagonalNormalEP",
    "MultivariateDiagonalNormalNP",
    "MultivariateDiagonalNormalVP",
    "MultivariateFixedVarianceNormalEP",
    "MultivariateFixedVarianceNormalNP",
    "MultivariateNormalEP",
    "MultivariateNormalNP",
    "MultivariateNormalVP",
    "MultivariateUnitVarianceNormalEP",
    "MultivariateUnitVarianceNormalNP",
    "NaturalParametrization",
    "NegativeBinomialEP",
    "NegativeBinomialNP",
    "NormalDP",
    "NormalEP",
    "NormalNP",
    "NormalVP",
    "PoissonEP",
    "PoissonNP",
    "RayleighEP",
    "RayleighNP",
    "RealField",
    "Ring",
    "Samplable",
    "ScalarSupport",
    "SimpleDistribution",
    "SimpleDistributionInfo",
    "SimplexSupport",
    "SoftplusNormalEP",
    "SoftplusNormalNP",
    "SquareMatrixSupport",
    "SubDistributionInfo",
    "SubsimplexSupport",
    "Support",
    "SymmetricMatrixSupport",
    "UnitVarianceLogNormalEP",
    "UnitVarianceLogNormalNP",
    "UnitVarianceNormalEP",
    "UnitVarianceNormalNP",
    "UnitVarianceSoftplusNormalEP",
    "UnitVarianceSoftplusNormalNP",
    "VectorSupport",
    "VonMisesFisherEP",
    "VonMisesFisherNP",
    "WeibullEP",
    "WeibullNP",
    "WishartEP",
    "WishartNP",
    "distribution_parameter",
    "expectation_parameters_from_characteristic_function",
    "flat_dict_of_observations",
    "flat_dict_of_parameters",
    "flatten_mapping",
    "parameter_dot_product",
    "parameter_holomorphic_dot",
    "parameter_map",
    "parameter_mean",
    "parameters",
    "unflatten_mapping",
]
