# Distributions

EFAX supports the following distributions, grouped by their support:

- normal:
    - univariate real:
        - with unit variance
        - with arbitrary parameters
    - univariate complex:
        - with unit variance and zero pseudo-variance
        - with arbitrary parameters
    - multivariate real:
        - with unit variance
        - with fixed variance
        - with isotropic variance
        - with diagonal variance
        - with arbitrary parameters
    - multivariate complex:
        - with unit variance and zero pseudo-variance
        - circularly symmetric
    - softplus-transformed:
        - with unit variance
        - with arbitrary parameters
    - log-normal (exponential-transformed):
        - with unit variance
        - with arbitrary parameters
- on a finite set:
    - Bernoulli
    - categorical
- on the nonnegative integers:
    - geometric
    - logarithmic
    - negative binomial
    - Poisson
- on the positive reals:
    - Rayleigh
    - Weibull
    - chi
    - chi-square
    - exponential
    - gamma
    - inverse Gaussian
    - inverse gamma
- on the simplex:
    - beta
    - Dirichlet
    - generalized Dirichlet
- on spheres:
    - von Mises-Fisher
    - complex von Mises
- on positive-definite matrices:
    - Wishart

To add a distribution, see the
[contribution guidelines](https://github.com/NeilGirdhar/efax/blob/main/CONTRIBUTING.md).
