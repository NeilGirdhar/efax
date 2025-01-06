from array_api_compat import array_namespace
from scipy.stats._distribution_infrastructure import \
    MonotonicTransformedDistribution  # noqa: PLC2701
from scipy.stats._distribution_infrastructure import ContinuousDistribution
from tjax import NumpyArray, inverse_softplus, softplus


def abs_derivative_inverse_softplus(y: NumpyArray, /, *, xp: object = None) -> NumpyArray:
    if xp is None:
        xp = array_namespace(y)
    return xp.abs(xp.reciprocal(-xp.expm1(-y)))  # pyright: ignore


def softplus_distribution(x: ContinuousDistribution, /) -> ContinuousDistribution:
    return MonotonicTransformedDistribution(x, g=softplus, h=inverse_softplus,
                                            dh=abs_derivative_inverse_softplus)
