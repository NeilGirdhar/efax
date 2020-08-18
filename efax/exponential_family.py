from abc import abstractmethod
from typing import Any, Tuple

from chex import Array
from ipromise import AbstractBaseClass
from jax import numpy as jnp
from tjax import RealArray, Shape, custom_jvp, jit, real_dtype

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

    # Magic methods --------------------------------------------------------------------------------
    def __init_subclass__(cls) -> None:
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
            method = jit(original_method, static_argnums=(0,))
            setattr(cls, f'_original_{name}', method)
            setattr(cls, name, method)

            if name != 'log_normalizer':
                continue

            method_jvp: Any = custom_jvp(method, nondiff_argnums=(0,))

            def ln_jvp(self: ExponentialFamily,
                       primals: Tuple[Array],
                       tangents: Tuple[Array]) -> Tuple[Array, Array]:
                q, = primals
                q_dot, = tangents
                y = self.log_normalizer(q)
                y_dot = jnp.sum(self.nat_to_exp(q) * q_dot, axis=-1)
                return y, y_dot

            method_jvp.defjvp(ln_jvp)

            # def method_fwd(self: ExponentialFamily, q: Array) -> Tuple[Array, Tuple[Array]]:
            #     return self.log_normalizer(q), (self.nat_to_exp(q),)
            #
            # def method_bwd(self: ExponentialFamily,
            #                residuals: Tuple[Array],
            #                y_bar: Array) -> Tuple[Array]:
            #     exp_parameters, = residuals
            #     return (y_bar * exp_parameters,)
            #
            # method.defvjp(method_fwd, method_bwd)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.num_parameters == other.num_parameters
                and self.shape == other.shape
                and self.observation_shape == other.observation_shape)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.num_parameters,
                     self.shape,
                     self.observation_shape))

    # Abstract methods -----------------------------------------------------------------------------
    @abstractmethod
    def log_normalizer(self, q: Array) -> RealArray:
        """
        Args:
            q: The natural parameters having shape self.shape_including_parameters().
        Returns: The log-normalizer having shape self.shape.
        """
        raise NotImplementedError

    @abstractmethod
    def nat_to_exp(self, q: Array) -> Array:
        """
        Args:
            q: The natural parameters having shape self.shape_including_parameters().
        Returns: The corresponding expectation parameters having shape
            self.shape_including_parameters().
        """
        raise NotImplementedError

    @abstractmethod
    def exp_to_nat(self, p: Array) -> Array:
        """
        Args:
            q: The expectation parameters having shape self.shape_including_parameters().
        Returns: The corresponding natural parameters having shape
            self.shape_including_parameters().
        """
        raise NotImplementedError

    @abstractmethod
    def sufficient_statistics(self, x: Array) -> Array:
        """
        Args:
            x: The sample having shape self.shape_including_observations().
        Returns: The corresponding sufficient statistics having shape
            self.shape_including_parameters().
        """
        raise NotImplementedError

    # New methods ----------------------------------------------------------------------------------
    def shape_including_parameters(self) -> Shape:
        return (*self.shape, self.num_parameters)

    def shape_including_observations(self) -> Shape:
        return (*self.shape, *self.observation_shape)

    def cross_entropy(self, p: Array, q: Array) -> RealArray:
        """
        Args:
            p: The expectation parameters of the observation having shape
                self.shape_including_parameters().
            q: The natural parameters of the prediction having shape
                self.shape_including_parameters().
        Returns:
            The cross entropy having shape self.shape.
        """
        p_dot_q = jnp.sum(p * q, axis=-1).real
        return (-p_dot_q
                + self.log_normalizer(q)
                - self.expected_carrier_measure(p))

    def entropy(self, q: Array) -> RealArray:
        """
        Args:
            q: The natural parameters of the prediction having shape
                self.shape_including_parameters().
        Returns: The entropy having shape self.shape.
        """
        return self.cross_entropy(self.nat_to_exp(q), q)

    def scaled_cross_entropy(
            self, k: real_dtype, kp: Array, q: Array) -> RealArray:
        """
        Args:
            k: The scale.
            kp: The expectation parameters of the observation times k, having shape
                self.shape_including_parameters().
            q: The natural parameters of the prediction having shape
                self.shape_including_parameters().
        Returns: k * self.cross_entropy(kp / k, q) having shape self.shape.

        Avoids division when the expected_carrier_measure is zero.
        """
        return (-jnp.real(jnp.sum(kp * q, axis=-1))
                + k * (self.log_normalizer(q)
                       - self.expected_carrier_measure(kp / k)))

    def carrier_measure(self, x: Array) -> RealArray:
        """
        Args:
            x: The sample having shape self.shape_including_observations().
        Returns: The corresponding carrier measure having shape self.shape.
        """
        shape = x.shape[: len(x.shape) - len(self.observation_shape)]
        return jnp.zeros(shape)

    def expected_carrier_measure(self, p: Array) -> RealArray:
        """
        Args:
            p: The expectation parameters of a distribution having shape
                self.shape_including_parameters().
        Returns: The expected carrier measure of the distribution having shape self.shape.  This is
            the missing term from the inner product between the observed distribution and the
            predicted distribution.
        """
        # pylint: disable=unused-argument
        shape = p.shape[: -1]
        return jnp.zeros(shape)

    def pdf(self, q: Array, x: Array) -> RealArray:
        """
        Args:
            q: The natural parameters of a distribution having shape
                self.shape_including_parameters().
            x: The sample having shape self.shape_including_observations().
        Returns: The distribution's density or mass function at x having shape self.shape.
        """
        tx = self.sufficient_statistics(x)
        tx_dot_q = jnp.real(jnp.sum(tx * q, axis=-1))
        return jnp.exp(tx_dot_q
                       - self.log_normalizer(q)
                       + self.carrier_measure(x))
