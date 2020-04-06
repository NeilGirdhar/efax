from abc import abstractmethod

import jax
import jax.numpy as jnp
from ipromise import AbstractBaseClass

from .tensors import RealTensor, Shape, Tensor, real_dtype

__all__ = ['ExponentialFamily']


class ExponentialFamily(AbstractBaseClass):

    """
    An Exponential family distribution.
    """

    def __init__(self,
                 *,
                 num_parameters: int,
                 shape: Shape = (),
                 observation_shape: Shape = ()):
        """
        Args:
            num_parameters: The number of parameters of the exponential family.
            shape: The shape of this object.
            observation_shape: The shape of an observation.
        """
        self.num_parameters = num_parameters
        self.shape = shape
        self.observation_shape = observation_shape

    # Magic methods -----------------------------------------------------------
    def __init_subclass__(cls):
        super().__init_subclass__()
        if cls.__name__ in ['VonMises', 'VonMisesFisher']:
            return

        # Apply jit.
        for name in ['log_normalizer',
                     'nat_to_exp',
                     'sufficient_statistics',
                     'cross_entropy',
                     'entropy',
                     'scaled_cross_entropy',
                     'carrier_measure',
                     'expected_carrier_measure',
                     'pdf']:
            original_method = getattr(cls, name)
            method = jax.jit(original_method, static_argnums=(0,))
            if name == 'log_normalizer':
                method = jax.custom_jvp(method, nondiff_argnums=(0,))
            setattr(cls, f'_original_{name}', method)
            setattr(cls, name, method)

            if name != 'log_normalizer':
                continue

            # pylint: disable=unused-variable
            @method.defjvp
            def ln_jvp(self, primals, tangents):
                x, = primals
                x_dot, = tangents
                y = self.log_normalizer(x)
                y_dot = jnp.sum(self.nat_to_exp(x) * x_dot, axis=-1)
                return y, y_dot

    def __repr__(self):
        return f"{type(self).__name__}(shape={self.shape})"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.num_parameters == other.num_parameters
                and self.shape == other.shape
                and self.observation_shape == other.observation_shape)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.num_parameters,
                     self.shape,
                     self.observation_shape))

    # Abstract methods --------------------------------------------------------
    @abstractmethod
    def log_normalizer(self, q: Tensor) -> RealTensor:
        """
        Args:
            q:
                The natural parameters having shape
                self.shape_including_parameters().
        Returns:
            The log-normalizer having shape self.shape.
        """
        raise NotImplementedError

    @abstractmethod
    def nat_to_exp(self, q: Tensor) -> Tensor:
        """
        Args:
            q:
                The natural parameters having shape
                self.shape_including_parameters().
        Returns:
            The corresponding expectation parameters having shape
            self.shape_including_parameters().
        """
        raise NotImplementedError

    @abstractmethod
    def exp_to_nat(self, p: Tensor) -> Tensor:
        """
        Args:
            q:
                The expectation parameters having shape
                self.shape_including_parameters().
        Returns:
            The corresponding natural parameters having shape
            self.shape_including_parameters().
        """
        raise NotImplementedError

    @abstractmethod
    def sufficient_statistics(self, x: Tensor) -> Tensor:
        """
        Args:
            x: The sample having shape self.shape_including_observations().
        Returns:
            The corresponding sufficient statistics having shape
            self.shape_including_parameters().
        """
        raise NotImplementedError

    # New methods -------------------------------------------------------------
    def shape_including_parameters(self) -> Shape:
        return (*self.shape, self.num_parameters)

    def shape_including_observations(self) -> Shape:
        return (*self.shape, *self.observation_shape)

    def cross_entropy(self, p: Tensor, q: Tensor) -> RealTensor:
        """
        Args:
            p:
                The expectation parameters of the observation having shape
                self.shape_including_parameters().
            q:
                The natural parameters of the prediction having shape
                self.shape_including_parameters().
        Returns:
            The cross entropy having shape self.shape.
        """
        p_dot_q = jnp.sum(p * q, axis=-1).real
        return (-p_dot_q
                + self.log_normalizer(q)
                - self.expected_carrier_measure(p))

    def entropy(self, q: Tensor) -> RealTensor:
        """
        Args:
            q:
                The natural parameters of the prediction having shape
                self.shape_including_parameters().
        Returns:
            The entropy having shape self.shape.
        """
        return self.cross_entropy(self.nat_to_exp(q), q)

    def scaled_cross_entropy(
            self, k: real_dtype, kp: Tensor, q: Tensor) -> RealTensor:
        """
        Args:
            k: The scale.
            kp:
                The expectation parameters of the observation times k, having
                shape self.shape_including_parameters().
            q:
                The natural parameters of the prediction having shape
                self.shape_including_parameters().
        Returns:
            k * self.cross_entropy(kp / k, q)

            having shape self.shape.

        Avoids division when the expected_carrier_measure is zero.
        """
        return (-jnp.real(jnp.sum(kp * q, axis=-1))
                + k * (self.log_normalizer(q)
                       - self.expected_carrier_measure(kp / k)))

    def carrier_measure(self, x: Tensor) -> RealTensor:
        """
        Args:
            x: The sample having shape self.shape_including_observations().
        Returns:
            The corresponding carrier measure having shape self.shape.
        """
        return jnp.zeros(self.shape)

    def expected_carrier_measure(self, p: Tensor) -> RealTensor:
        """
        Args:
            p:
                The expectation parameters of a distribution having shape
                self.shape_including_parameters().
        Returns:
            The expected carrier measure of the distribution having shape
            self.shape.  This is the missing term from the inner product
            between the observed distribution and the predicted distribution.
        """
        # pylint: disable=unused-argument
        return jnp.zeros(self.shape)

    def pdf(self, q: Tensor, x: Tensor) -> RealTensor:
        """
        Args:
            q:
                The natural parameters of a distribution having shape
                self.shape_including_parameters().
            x: The sample having shape self.shape_including_observations().
        Returns:
            The distribution's density or mass function at x having shape
            self.shape.
        """
        tx = self.sufficient_statistics(x)
        tx_dot_q = jnp.real(jnp.sum(tx * q, axis=-1))
        return jnp.exp(
            tx_dot_q
            - self.log_normalizer(q)
            + self.carrier_measure(x))
