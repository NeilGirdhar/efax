from jax import grad, jit
from numpy.testing import assert_allclose

from ..complex_normal import ComplexNormal
from ..von_mises import VonMises

# todo: Block VonMises until https://github.com/google/jax/issues/2466 is
# resolved.
# todo: Block ComplexNormal until https://github.com/google/jax/issues/2488 is
# resolved.


def test_conversion(generator, distribution_info):
    """
    Test that the conversion between the different parametrizations are
    consistent.
    """
    if isinstance(distribution_info.my_distribution, VonMises):
        return
    my_distribution = distribution_info.my_distribution

    for _ in range(10):
        parameters = distribution_info.exp_parameter_generator(
            generator, shape=(4, 3))
        nat_parameters = my_distribution.exp_to_nat(parameters)
        exp_parameters = my_distribution.nat_to_exp(nat_parameters)
        assert_allclose(parameters, exp_parameters, rtol=1e-4)


def test_scaled_cross_entropy(generator, distribution_info):
    """
    Test that the “scaled cross entropy” matches the cross entropy, scaled.
    """
    if isinstance(distribution_info.my_distribution, VonMises):
        return
    k = 0.3
    p = distribution_info.exp_parameter_generator(generator, shape=())
    q = distribution_info.nat_parameter_generator(generator, shape=())
    my_distribution = distribution_info.my_distribution
    try:
        assert_allclose(
            my_distribution.scaled_cross_entropy(k, k * p, q),
            k * my_distribution.cross_entropy(p, q),
            rtol=3e-6)
    except NotImplementedError:
        pass


def test_gradient_log_normalizer(generator, distribution_info):
    """
    Tests that the gradient log-normalizer evaluates to the same as the
    gradient of the log-normalizer.
    """
    if isinstance(distribution_info.my_distribution, VonMises):
        return
    if isinstance(distribution_info.my_distribution, ComplexNormal):
        return
    my_distribution = distribution_info.my_distribution

    def original_f(q):
        # pylint: disable=protected-access
        return my_distribution._original_log_normalizer(q)

    def f(q):
        return my_distribution.log_normalizer(q)

    original_gln = jit(grad(original_f))
    gln = jit(grad(f))
    nat_to_exp = my_distribution.nat_to_exp

    for _ in range(20):
        nat_parameters = distribution_info.nat_parameter_generator(
            generator,
            shape=())

        original_gln_x = original_gln(nat_parameters)
        gln_x = gln(nat_parameters)
        exp_parameters = nat_to_exp(nat_parameters)

        assert_allclose(original_gln_x,
                        exp_parameters,
                        rtol=1e-5)
        assert_allclose(gln_x,
                        exp_parameters,
                        rtol=1e-5)
