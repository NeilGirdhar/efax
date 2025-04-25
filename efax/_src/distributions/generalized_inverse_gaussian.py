from __future__ import annotations

import jax
from array_api_compat import array_namespace
from tjax import JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override
from logbesselk.jax import log_bessel_k, bessel_kratio

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..natural_parametrization import NaturalParametrization
from ..parameter import (RealField, ScalarSupport, Support, distribution_parameter,
                        negative_support, positive_support)
from ..parametrization import SimpleDistribution


@dataclass
class GeneralizedInverseGaussianNP(HasEntropyNP['GeneralizedInverseGaussianEP'],
                                   Samplable,
                                   NaturalParametrization['GeneralizedInverseGaussianEP', JaxRealArray],
                                   SimpleDistribution):
    """The natural parametrization of the generalized inverse Gaussian distribution.

    Args:
        p_minus_one: p - 1, where p is the shape parameter
        negative_a_over_two: -a/2, where a is the first scale parameter
        negative_b_over_two: -b/2, where b is the second scale parameter
    """
    p_minus_one: JaxRealArray = distribution_parameter(ScalarSupport())
    negative_a_over_two: JaxRealArray = distribution_parameter(ScalarSupport(ring=negative_support))
    negative_b_over_two: JaxRealArray = distribution_parameter(ScalarSupport(ring=negative_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.p_minus_one.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=positive_support)

    @override
    def log_normalizer(self) -> JaxRealArray:
        xp = self.array_namespace()
        p = self.p_minus_one + 1.0
        eta2 = self.negative_a_over_two
        eta3 = self.negative_b_over_two
        
        # Calculate sqrt(4*eta2*eta3)
        arg = xp.sqrt(4.0 * eta2 * eta3)
        
        # Calculate log[2*K_p(arg)]
        log_2k = xp.log(2.0) + log_bessel_k(p, arg)
        
        # Calculate -(p/2)*log(eta2/eta3)
        log_ratio_term = -0.5 * p * (xp.log(-eta2) - xp.log(-eta3))
        
        return log_2k + log_ratio_term

    @override
    def to_exp(self) -> GeneralizedInverseGaussianEP:
        """Convert from natural to expectation parameters."""
        xp = self.array_namespace()
        p = self.p_minus_one + 1.0
        eta2 = self.negative_a_over_two
        eta3 = self.negative_b_over_two
        
        # Calculate sqrt(4*eta2*eta3)
        arg = xp.sqrt(4.0 * eta2 * eta3)
        
        # Calculate E[ln x]
        log_eta_ratio = xp.log(-eta2) - xp.log(-eta3)
        dlogk_dp = jax.grad(log_bessel_k, 0)
        mean_log = -0.5 * log_eta_ratio + dlogk_dp(p, arg)
        
        # Calculate E[x]
        kratio_value = bessel_kratio(p, arg, d=1)
        mean = xp.sqrt(-eta3 / eta2) * kratio_value
        
        # Calculate E[1/x]
        mean_inv = xp.sqrt(-eta2 / eta3) * kratio_value + (p / (-eta3))
        
        return GeneralizedInverseGaussianEP(mean_log, mean, mean_inv)

    @override
    def carrier_measure(self, x: JaxRealArray) -> JaxRealArray:
        """The carrier measure for GIG is h(x) = 1, so log(h(x)) = 0."""
        xp = self.array_namespace(x)
        return xp.zeros(x.shape)

    @override
    @classmethod
    def sufficient_statistics(cls, x: JaxRealArray, **fixed_parameters: JaxArray
                             ) -> GeneralizedInverseGaussianEP:
        """Compute the sufficient statistics T(x) = (ln x, x, 1/x)."""
        xp = array_namespace(x)
        log_x = xp.log(x)
        inv_x = xp.reciprocal(x)
        return GeneralizedInverseGaussianEP(log_x, x, inv_x)


@dataclass
class GeneralizedInverseGaussianEP(HasEntropyEP[GeneralizedInverseGaussianNP],
                                   ExpectationParametrization[GeneralizedInverseGaussianNP],
                                   Samplable,
                                   SimpleDistribution):
    """The expectation parametrization of the generalized inverse Gaussian distribution.

    Args:
        mean_log: E[ln x]
        mean: E[x]
        mean_inv: E[1/x]
    """
    mean_log: JaxRealArray = distribution_parameter(ScalarSupport())
    mean: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))
    mean_inv: JaxRealArray = distribution_parameter(ScalarSupport(ring=positive_support))

    @property
    @override
    def shape(self) -> Shape:
        return self.mean.shape

    @override
    @classmethod
    def domain_support(cls) -> ScalarSupport:
        return ScalarSupport(ring=positive_support)

    @classmethod
    @override
    def natural_parametrization_cls(cls) -> type[GeneralizedInverseGaussianNP]:
        return GeneralizedInverseGaussianNP
    
    @override
    def expected_carrier_measure(self) -> JaxRealArray:
        xp = self.array_namespace()
        return xp.zeros(self.shape)
