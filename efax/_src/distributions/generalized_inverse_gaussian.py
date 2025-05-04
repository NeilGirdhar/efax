from __future__ import annotations

import jax
import jax.lax as lax
import jax.random as jr
import jax.numpy as jnp
from array_api_compat import array_namespace
from tjax import JaxArray, JaxRealArray, KeyArray, Shape
from tjax.dataclasses import dataclass
from typing_extensions import override, Tuple
from logbesselk.jax import log_bessel_k, bessel_kratio

from ..expectation_parametrization import ExpectationParametrization
from ..interfaces.samplable import Samplable
from ..mixins.has_entropy import HasEntropyEP, HasEntropyNP
from ..mixins.exp_to_nat.exp_to_nat import ExpToNat
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
        # Convert natural parameters to original parameters
        p = self.p_minus_one + 1.0
        a = -2.0 * self.negative_a_over_two
        b = -2.0 * self.negative_b_over_two

        y = xp.sqrt(b / a)
        z = xp.sqrt(a * b)
        
        if p.ndim == 0:
            log_2k = xp.log(2.0) + log_bessel_k(p, z)
        else:
            log_k = jax.jit(jax.vmap(log_bessel_k, 0))
            log_2k = xp.log(2.0) + log_k(p, z)
        log_ratio_term = p * xp.log(y)
        return log_2k + log_ratio_term

    @override
    def to_exp(self) -> GeneralizedInverseGaussianEP:
        """Convert from natural to expectation parameters."""
        xp = self.array_namespace()
        
        # Convert natural parameters to original parameters
        p = self.p_minus_one + 1.0
        a = -2.0 * self.negative_a_over_two
        b = -2.0 * self.negative_b_over_two

        y = xp.sqrt(b / a)
        z = xp.sqrt(a * b)

        # TBD: Use naive finite difference for dlogk_dp for now
        # as jax.grad does not work with log_bessel_k function
        # Issue: https://github.com/tk2lab/logbesselk/issues/33
        # mean_log = xp.log(y) + dlogk_dp(p, z)
        # eps = 1e-6 would cause numerical issue in log_bessel_k
        if p.ndim == 0:
            eps = 1e-5
            kratio = xp.exp(log_bessel_k(p + 1, z) - log_bessel_k(p, z))
            dlogk_dp = (log_bessel_k(p + eps, z) - log_bessel_k(p - eps, z)) / (2.0 * eps)
        else:
            eps = xp.ones_like(p) * 1e-5
            logk = jax.jit(jax.vmap(log_bessel_k, 0))
            kratio = xp.exp(logk(p + xp.ones_like(p), z) - logk(p, z))
            dlogk_dp = (logk(p + eps, z) - logk(p - eps, z)) / (2.0 * eps)
        
        mean_log = xp.log(y) + dlogk_dp
        mean = kratio * y
        mean_inv = (1.0 / y) * kratio - 2.0 * p / b
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
    
    @override
    def sample(self, key: KeyArray, shape: Shape, burnin: int = 100) -> JaxRealArray:
        """Sample from the Generalized Inverse Gaussian distribution using Gibbs sampling.
        
        References:
            Peña, V., & Jauch, M. (2025). Properties of the generalized inverse Gaussian with applications to Monte Carlo simulation and distribution function evaluation. Statistics & Probability Letters, 110359.
            https://github.com/michaeljauch/gig/blob/main/GIG_Gibbs_Benchmark.R
        """
        xp = self.array_namespace()
        
        # Convert natural parameters to original parameters
        p = self.p_minus_one + 1.0
        a = -2.0 * self.negative_a_over_two
        b = -2.0 * self.negative_b_over_two
        
        # Set up shape dimensions
        full_shape = self.shape if shape is None else shape + self.shape
        
        def gibbs_update(carry: Tuple[JaxRealArray, KeyArray], _):
            x_prev, subkey = carry
            subkey, gamma_key, ig_key = jr.split(subkey, 3)
            
            # For p < -0.5 case
            alpha_neg = -(p + 0.5)
            beta_neg = x_prev
            y_neg = jr.gamma(gamma_key, alpha_neg, shape=x_prev.shape)
            y_neg = y_neg / beta_neg
            mu_neg = xp.sqrt(b / (a + 2.0 * y_neg))
            lambda_neg = b
            
            # For p >= -0.5 case
            alpha_pos = p + 0.5
            beta_pos = 1.0 / x_prev
            y_pos = jr.gamma(gamma_key, alpha_pos, shape=x_prev.shape)
            y_pos = y_pos / beta_pos
            b2y_pos = b + 2.0 * y_pos
            mu_pos = xp.sqrt(b2y_pos / a)
            lambda_pos = b2y_pos
            
            # Select parameters based on p value
            mu = xp.where(p < -0.5, mu_neg, mu_pos)
            lambda_ = xp.where(p < -0.5, lambda_neg, lambda_pos)
            
            # Sample from inverse Gaussian using selected parameters
            # Based on InverseGaussianNP.sample implementation
            nu = jr.normal(ig_key, x_prev.shape)
            u = jr.uniform(jr.split(ig_key)[0], x_prev.shape)
            y = xp.square(nu)
            x_ig = mu + 0.5 * xp.square(mu) / lambda_ * y - (
                mu / (2.0 * lambda_) * xp.sqrt(4.0 * mu * lambda_ * y + 
                                              xp.square(mu) * xp.square(y)))

            # Apply the acceptance criterion
            x_new = xp.where(u <= mu / (mu + x_ig), x_ig, xp.square(mu) / x_ig)
            return (x_new, subkey), x_new
        
        # Run burn-in iterations
        carry = (xp.ones(self.shape), key)
        (x, key), _ = lax.scan(gibbs_update, carry, xp.arange(burnin))
        
        # Run actual sampling iterations
        num_samples = jnp.prod(jnp.array(shape)) if shape else 1
        _, samples = lax.scan(gibbs_update, (x, key), xp.arange(num_samples))
        return samples.reshape(full_shape)


@dataclass
class GeneralizedInverseGaussianEP(HasEntropyEP[GeneralizedInverseGaussianNP],
                                   ExpToNat[GeneralizedInverseGaussianNP],
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
    
    @override
    def sample(self, key: KeyArray, shape: Shape | None = None) -> JaxRealArray:
        return self.to_nat().sample(key, shape)
