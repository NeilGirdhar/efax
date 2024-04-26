"""The EFAX Library."""
from ._src.distributions.bernoulli import BernoulliEP, BernoulliNP
from ._src.distributions.beta import BetaEP, BetaNP
from ._src.distributions.chi import ChiEP, ChiNP
from ._src.distributions.chi_square import ChiSquareEP, ChiSquareNP
from ._src.distributions.cmvn.circularly_symmetric import (ComplexCircularlySymmetricNormalEP,
                                                           ComplexCircularlySymmetricNormalNP)
from ._src.distributions.cmvn.unit import (ComplexMultivariateUnitNormalEP,
                                           ComplexMultivariateUnitNormalNP)
from ._src.distributions.complex_normal.complex_normal import ComplexNormalEP, ComplexNormalNP
from ._src.distributions.complex_normal.unit import ComplexUnitNormalEP, ComplexUnitNormalNP
from ._src.distributions.dirichlet import DirichletEP, DirichletNP
from ._src.distributions.exponential import ExponentialEP, ExponentialNP
from ._src.distributions.gamma import GammaEP, GammaNP, GammaVP
from ._src.distributions.gen_dirichlet import GeneralizedDirichletEP, GeneralizedDirichletNP
from ._src.distributions.geometric import GeometricEP, GeometricNP
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
from ._src.distributions.multivariate_normal.unit import (MultivariateUnitNormalEP,
                                                          MultivariateUnitNormalNP)
from ._src.distributions.negative_binomial import NegativeBinomialEP, NegativeBinomialNP
from ._src.distributions.normal.normal import NormalEP, NormalNP, NormalVP
from ._src.distributions.normal.unit import UnitNormalEP, UnitNormalNP
from ._src.distributions.poisson import PoissonEP, PoissonNP
from ._src.distributions.rayleigh import RayleighEP, RayleighNP
from ._src.distributions.von_mises import VonMisesFisherEP, VonMisesFisherNP
from ._src.distributions.weibull import WeibullEP, WeibullNP
from ._src.expectation_parametrization import ExpectationParametrization
from ._src.interfaces.conjugate_prior import HasConjugatePrior, HasGeneralizedConjugatePrior
from ._src.interfaces.multidimensional import Multidimensional
from ._src.interfaces.samplable import Samplable
from ._src.mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ._src.natural_parametrization import NaturalParametrization
from ._src.parameter import (BooleanRing, ComplexField, IntegralRing, RealField, Ring,
                             ScalarSupport, SquareMatrixSupport, Support, SymmetricMatrixSupport,
                             VectorSupport)
from ._src.parametrization import Parametrization
from ._src.scipy_replacement.complex_multivariate_normal import ScipyComplexMultivariateNormal
from ._src.scipy_replacement.complex_normal import ScipyComplexNormal
from ._src.scipy_replacement.dirichlet import ScipyDirichlet, ScipyGeneralizedDirichlet
from ._src.scipy_replacement.multivariate_normal import ScipyMultivariateNormal
from ._src.scipy_replacement.von_mises import ScipyVonMises
from ._src.tools import parameter_dot_product, parameter_map, parameter_mean

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
    'ComplexMultivariateUnitNormalEP',
    'ComplexMultivariateUnitNormalNP',
    'ComplexNormalEP',
    'ComplexNormalNP',
    'ComplexUnitNormalEP',
    'ComplexUnitNormalNP',
    'DirichletEP',
    'DirichletNP',
    'ExpectationParametrization',
    'ExponentialEP',
    'ExponentialNP',
    'GammaEP',
    'GammaNP',
    'GammaVP',
    'GeneralizedDirichletEP',
    'GeneralizedDirichletNP',
    'GeometricEP',
    'GeometricNP',
    'HasConjugatePrior',
    'HasEntropyEP',
    'HasEntropyNP',
    'HasGeneralizedConjugatePrior',
    'IntegralRing',
    'IsotropicNormalEP',
    'IsotropicNormalNP',
    'LogarithmicEP',
    'LogarithmicNP',
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
    'MultivariateUnitNormalEP',
    'MultivariateUnitNormalNP',
    'NaturalParametrization',
    'NegativeBinomialEP',
    'NegativeBinomialNP',
    'NormalEP',
    'NormalNP',
    'NormalVP',
    'Parametrization',
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
    'ScipyMultivariateNormal',
    'ScipyVonMises',
    'SquareMatrixSupport',
    'Support',
    'SymmetricMatrixSupport',
    'UnitNormalEP',
    'UnitNormalNP',
    'VectorSupport',
    'VonMisesFisherEP',
    'VonMisesFisherNP',
    'WeibullEP',
    'WeibullNP',
    'parameter_dot_product',
    'parameter_map',
    'parameter_mean',
]
