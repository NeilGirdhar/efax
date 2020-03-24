# pylint: disable=protected-access
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

from . import ScipyComplexMultivariateNormal, ScipyComplexNormal


@pytest.fixture(scope='module')
def set_numpy_state():
    with np.printoptions(precision=4, suppress=True):
        yield


def random_complex(generator):
    return sum(x ** 0.5 * generator.normal()
               for x in [0.5, 0.5j])


def build_uvcn(generator):
    mean = random_complex(generator)
    variance = generator.exponential()
    pseudo_variance = (
        variance
        * generator.beta(2, 2)
        * np.exp(1j * generator.uniform(0, 2 * np.pi)))
    return ScipyComplexNormal(mean, variance, pseudo_variance)


def build_mvcn(generator, size=2, polarization=0.98, regularization=0.01):
    DIRECTIONS = 3

    def random_vector(number_of_vectors=()):
        return sum(x * generator.multivariate_normal(
            np.zeros(size),
            cov=np.eye(size),
            size=number_of_vectors)
            for x in [1, 1j])

    mean = random_vector()
    z = random_vector(DIRECTIONS)
    weights = np.array(range(DIRECTIONS)) + 1.5

    variance = (np.average(z.conj()[:, np.newaxis, :] * z[..., np.newaxis],
                           weights=weights,
                           axis=0)
                + regularization * np.eye(size))
    pseudo_variance = (np.average(z[:, np.newaxis, :] * z[..., np.newaxis],
                                  weights=weights,
                                  axis=0) * polarization
                       + regularization * np.eye(size))

    return ScipyComplexMultivariateNormal(
        mean, variance, pseudo_variance)


def test_univariate_pdf(generator):
    dist = build_uvcn(generator)
    mvn = multivariate_normal(mean=dist._multivariate_normal_mean(),
                              cov=dist._multivariate_normal_cov())
    x = random_complex(generator)
    assert_allclose(dist.pdf(x),
                    mvn.pdf(np.array([x.real, x.imag])))


def test_univariate_rvs(generator):
    dist = build_uvcn(generator)

    size = (200, 100)
    rvs = dist.rvs(random_state=generator, size=size)

    assert rvs.shape == size

    estimated_mean = np.average(rvs)

    centered_rvs = rvs - dist.mean

    estimated_variance = np.average(
        centered_rvs.conj() * centered_rvs).real
    estimated_pseudo_variance = np.average(np.square(centered_rvs))

    assert_allclose(estimated_mean, dist.mean, rtol=1e-2)
    assert_allclose(estimated_variance, dist.variance, rtol=1e-2)
    assert_allclose(estimated_pseudo_variance, dist.pseudo_variance,
                    rtol=1e-2, atol=1e-2)


def test_multivariate_rvs(generator):
    dist = build_mvcn(generator)

    size = (1800, 1700)
    axis = tuple(range(2))
    rvs = dist.rvs(random_state=generator, size=size)

    assert rvs.shape == size + (dist.size,)

    estimated_mean = np.average(rvs, axis=axis)

    centered_rvs = rvs - dist.mean

    estimated_variance = np.average(
        centered_rvs[..., np.newaxis]
        * centered_rvs.conj()[..., np.newaxis, :],
        axis=axis)
    estimated_pseudo_variance = np.average(
        centered_rvs[..., np.newaxis]
        * centered_rvs[..., np.newaxis, :],
        axis=axis)

    assert_allclose(estimated_mean, dist.mean, rtol=1e-2, atol=1e-2)
    assert_allclose(estimated_variance, dist.variance,
                    rtol=2e-1, atol=1e-1)
    assert_allclose(estimated_pseudo_variance, dist.pseudo_variance,
                    rtol=2e-1, atol=1e-1)


@pytest.mark.parametrize('polarization', [0.5, 1.0])
def test_p_hermitian(generator, polarization):
    dist = build_mvcn(generator, polarization=polarization)
    _, p_c = dist._r_and_p_c()
    assert_allclose(p_c, p_c.T.conj(), rtol=1e-2, atol=1e-2)
    eigenvalues = np.linalg.eigvals(p_c)
    assert np.all(eigenvalues.real >= 0)
    assert_allclose(eigenvalues.imag, 0.0, atol=1e-5)


def test_multivariate_conversion(generator):
    dist = build_mvcn(generator, polarization=0.5)

    eta, precision, pseudo_precision = dist.natural_parameters()
    mu, s, u = dist.natural_to_sample(eta, precision, pseudo_precision)

    assert_allclose(dist.mean, mu, rtol=1e-2, atol=1e-2)
    assert_allclose(dist.variance, s, rtol=1e-2, atol=1e-2)
    assert_allclose(dist.pseudo_variance, u, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_multivariate_pdf(generator, n):
    dist = build_mvcn(generator, polarization=0.5, size=n)
    mvn = multivariate_normal(mean=dist._multivariate_normal_mean(),
                              cov=dist._multivariate_normal_cov())
    x = np.array([random_complex(generator)
                  for _ in range(n)])
    xx = np.concatenate([x.real, x.imag], axis=-1)
    assert_allclose(0.5 * (dist.variance + dist.pseudo_variance).real,
                    mvn.cov[:n, :n])
    assert_allclose(0.5 * (dist.variance - dist.pseudo_variance).real,
                    mvn.cov[n:, n:])
    assert_allclose(0.5 * (dist.variance + dist.pseudo_variance).imag,
                    mvn.cov[n:, :n])
    assert_allclose(0.5 * (-dist.variance + dist.pseudo_variance).imag,
                    mvn.cov[:n, n:])

    assert_allclose(dist.pdf(np.zeros(n)), mvn.pdf(np.zeros(2 * n)),
                    atol=1e-3)
    assert_allclose(dist.pdf(x), mvn.pdf(xx))


def test_univariate_multivariate_consistency(generator):
    mv = build_mvcn(generator, size=1, polarization=0.5)
    uv = ScipyComplexNormal(mv.mean[0],
                            mv.variance[0, 0].real,
                            mv.pseudo_variance[0, 0])
    m_eta, m_h, m_j = mv.natural_parameters()
    u_eta, u_h, u_j = uv.natural_parameters()
    assert_allclose(m_eta, u_eta)
    assert_allclose(m_h, u_h)
    assert_allclose(m_j, u_j)

    x = random_complex(generator)
    assert_allclose(mv.pdf(np.array([x])), uv.pdf(x))
