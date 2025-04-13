"""The EFAX Library."""
from ._src.distributions.bernoulli import BernoulliEP, BernoulliNP
from ._src.distributions.beta import BetaEP, BetaNP
from ._src.distributions.chi import ChiEP, ChiNP
from ._src.distributions.chi_square import ChiSquareEP, ChiSquareNP
from ._src.distributions.cmvn.circularly_symmetric import (ComplexCircularlySymmetricNormalEP,
                                                           ComplexCircularlySymmetricNormalNP)
from ._src.distributions.cmvn.unit_variance import (ComplexMultivariateUnitVarianceNormalEP,
                                                    ComplexMultivariateUnitVarianceNormalNP)
from ._src.distributions.complex_normal.complex_normal import ComplexNormalEP, ComplexNormalNP
from ._src.distributions.complex_normal.unit_variance import (ComplexUnitVarianceNormalEP,
                                                              ComplexUnitVarianceNormalNP)
from ._src.distributions.dirichlet import DirichletEP, DirichletNP
from ._src.distributions.exponential import ExponentialEP, ExponentialNP
from ._src.distributions.gamma import GammaEP, GammaNP, GammaVP
from ._src.distributions.gen_dirichlet import GeneralizedDirichletEP, GeneralizedDirichletNP
from ._src.distributions.geometric import GeometricEP, GeometricNP
from ._src.distributions.inverse_gamma import InverseGammaEP, InverseGammaNP
from ._src.distributions.inverse_gaussian import InverseGaussianEP, InverseGaussianNP
from ._src.distributions.log_normal.log_normal import LogNormalEP, LogNormalNP
from ._src.distributions.log_normal.unit_variance import (UnitVarianceLogNormalEP,
                                                          UnitVarianceLogNormalNP)
from ._src.distributions.logarithmic import LogarithmicEP, LogarithmicNP
from ._src.distributions.multinomial import MultinomialEP, MultinomialNP
from ._src.distributions.multivariate_normal.arbitrary import (MultivariateNormalEP,
                                                               MultivariateNormalNP,
                                                               MultivariateNormalVP)
from ._src.distributions.multivariate_normal.diagonal import (MultivariateDiagonalNormalEP,
                                                              MultivariateDiagonalNormalNP,
                                                              MultivariateDiagonalNormalVP)
from ._src.distributions.multivariate_normal.fixed_variance import (
    MultivariateFixedVarianceNormalEP, MultivariateFixedVarianceNormalNP)
from ._src.distributions.multivariate_normal.isotropic import IsotropicNormalEP, IsotropicNormalNP
from ._src.distributions.multivariate_normal.unit_variance import (MultivariateUnitVarianceNormalEP,
                                                                   MultivariateUnitVarianceNormalNP)
from ._src.distributions.negative_binomial import NegativeBinomialEP, NegativeBinomialNP
from ._src.distributions.normal.normal import NormalDP, NormalEP, NormalNP, NormalVP
from ._src.distributions.normal.unit_variance import UnitVarianceNormalEP, UnitVarianceNormalNP
from ._src.distributions.poisson import PoissonEP, PoissonNP
from ._src.distributions.rayleigh import RayleighEP, RayleighNP
from ._src.distributions.softplus_normal.softplus import SoftplusNormalEP, SoftplusNormalNP
from ._src.distributions.softplus_normal.unit_variance import (UnitVarianceSoftplusNormalEP,
                                                               UnitVarianceSoftplusNormalNP)
from ._src.distributions.von_mises import VonMisesFisherEP, VonMisesFisherNP
from ._src.distributions.weibull import WeibullEP, WeibullNP
from ._src.expectation_parametrization import ExpectationParametrization
from ._src.interfaces.conjugate_prior import HasConjugatePrior, HasGeneralizedConjugatePrior
from ._src.interfaces.multidimensional import Multidimensional
from ._src.interfaces.samplable import Samplable
from ._src.iteration import (flat_dict_of_observations, flat_dict_of_parameters, flatten_mapping,
                             parameters, support, unflatten_mapping)
from ._src.mixins.has_entropy import HasEntropy, HasEntropyEP, HasEntropyNP
from ._src.natural_parametrization import NaturalParametrization
from ._src.parameter import (BooleanRing, ComplexField, IntegralRing, RealField, Ring,
                             ScalarSupport, SquareMatrixSupport, Support, SymmetricMatrixSupport,
                             VectorSupport)
from ._src.parametrization import Distribution, SimpleDistribution
from ._src.scipy_replacement.complex_multivariate_normal import ScipyComplexMultivariateNormal
from ._src.scipy_replacement.complex_normal import ScipyComplexNormal
from ._src.scipy_replacement.dirichlet import ScipyDirichlet, ScipyGeneralizedDirichlet
from ._src.scipy_replacement.geometric import ScipyGeometric
from ._src.scipy_replacement.joint import ScipyJointDistribution
from ._src.scipy_replacement.log_normal import ScipyLogNormal
from ._src.scipy_replacement.multivariate_normal import ScipyMultivariateNormal
from ._src.scipy_replacement.softplus_normal import ScipySoftplusNormal
from ._src.scipy_replacement.von_mises import ScipyVonMises, ScipyVonMisesFisher
from ._src.structure import Flattener, MaximumLikelihoodEstimator, Structure, SubDistributionInfo
from ._src.tools import parameter_dot_product, parameter_map, parameter_mean
from ._src.transform.joint import JointDistribution, JointDistributionE, JointDistributionN

__all__ = [
    'BernoulliEP',
    'BernoulliNP',
    'BetaEP',
    'BetaNP',
    'BooleanRing',
    'ChiEP',
    'ChiNP',
    'ChiSquareEP',
    'ChiSquareNP',
    'ComplexCircularlySymmetricNormalEP',
    'ComplexCircularlySymmetricNormalNP',
    'ComplexField',
    'ComplexMultivariateUnitVarianceNormalEP',
    'ComplexMultivariateUnitVarianceNormalNP',
    'ComplexNormalEP',
    'ComplexNormalNP',
    'ComplexUnitVarianceNormalEP',
    'ComplexUnitVarianceNormalNP',
    'DirichletEP',
    'DirichletNP',
    'Distribution',
    'ExpectationParametrization',
    'ExponentialEP',
    'ExponentialNP',
    'Flattener',
    'GammaEP',
    'GammaNP',
    'GammaVP',
    'GeneralizedDirichletEP',
    'GeneralizedDirichletNP',
    'GeometricEP',
    'GeometricNP',
    'HasConjugatePrior',
    'HasEntropy',
    'HasEntropyEP',
    'HasEntropyNP',
    'HasGeneralizedConjugatePrior',
    'IntegralRing',
    'InverseGammaEP',
    'InverseGammaNP',
    'InverseGaussianEP',
    'InverseGaussianNP',
    'IsotropicNormalEP',
    'IsotropicNormalNP',
    'JointDistribution',
    'JointDistributionE',
    'JointDistributionN',
    'LogNormalEP',
    'LogNormalNP',
    'LogarithmicEP',
    'LogarithmicNP',
    'MaximumLikelihoodEstimator',
    'Multidimensional',
    'MultinomialEP',
    'MultinomialNP',
    'MultivariateDiagonalNormalEP',
    'MultivariateDiagonalNormalNP',
    'MultivariateDiagonalNormalVP',
    'MultivariateFixedVarianceNormalEP',
    'MultivariateFixedVarianceNormalNP',
    'MultivariateNormalEP',
    'MultivariateNormalNP',
    'MultivariateNormalVP',
    'MultivariateUnitVarianceNormalEP',
    'MultivariateUnitVarianceNormalNP',
    'NaturalParametrization',
    'NegativeBinomialEP',
    'NegativeBinomialNP',
    'NormalDP',
    'NormalEP',
    'NormalNP',
    'NormalVP',
    'PoissonEP',
    'PoissonNP',
    'RayleighEP',
    'RayleighNP',
    'RealField',
    'Ring',
    'Samplable',
    'ScalarSupport',
    'ScipyComplexMultivariateNormal',
    'ScipyComplexNormal',
    'ScipyDirichlet',
    'ScipyGeneralizedDirichlet',
    'ScipyGeometric',
    'ScipyJointDistribution',
    'ScipyLogNormal',
    'ScipyMultivariateNormal',
    'ScipySoftplusNormal',
    'ScipyVonMises',
    'ScipyVonMisesFisher',
    'SimpleDistribution',
    'SoftplusNormalEP',
    'SoftplusNormalNP',
    'SquareMatrixSupport',
    'Structure',
    'SubDistributionInfo',
    'Support',
    'SymmetricMatrixSupport',
    'UnitVarianceLogNormalEP',
    'UnitVarianceLogNormalNP',
    'UnitVarianceNormalEP',
    'UnitVarianceNormalNP',
    'UnitVarianceSoftplusNormalEP',
    'UnitVarianceSoftplusNormalNP',
    'VectorSupport',
    'VonMisesFisherEP',
    'VonMisesFisherNP',
    'WeibullEP',
    'WeibullNP',
    'flat_dict_of_observations',
    'flat_dict_of_parameters',
    'flatten_mapping',
    'parameter_dot_product',
    'parameter_map',
    'parameter_mean',
    'parameters',
    'support',
    'unflatten_mapping',
]
