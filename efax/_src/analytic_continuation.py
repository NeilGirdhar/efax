from __future__ import annotations

from array_api_compat import array_namespace
from tjax import JaxArray, JaxComplexArray, Shape
from tjax.dataclasses import dataclass


@dataclass
class ComplexContinuation:
    """Analytic continuation of a complex field's real and imaginary coordinates."""

    real: JaxComplexArray
    imag: JaxComplexArray

    @property
    def shape(self) -> Shape:
        return self.real.shape

    @property
    def dtype(self) -> object:
        xp = array_namespace(self.real, self.imag)
        return xp.result_type(self.real, self.imag)

    def __array_namespace__(self, *, api_version: str | None = None) -> object:  # noqa: PLW3201
        return array_namespace(self.real, self.imag, api_version=api_version)

    def __neg__(self) -> ComplexContinuation:
        return ComplexContinuation(-self.real, -self.imag)

    def __add__(self, other: JaxArray | ComplexContinuation) -> ComplexContinuation:
        other_real, other_imag = complex_parts(other)
        return ComplexContinuation(self.real + other_real, self.imag + other_imag)

    def __radd__(self, other: JaxArray | ComplexContinuation) -> ComplexContinuation:
        return self + other

    def __sub__(self, other: JaxArray | ComplexContinuation) -> ComplexContinuation:
        other_real, other_imag = complex_parts(other)
        return ComplexContinuation(self.real - other_real, self.imag - other_imag)

    def __rsub__(self, other: JaxArray | ComplexContinuation) -> ComplexContinuation:
        other_real, other_imag = complex_parts(other)
        return ComplexContinuation(other_real - self.real, other_imag - self.imag)

    def __mul__(self, other: JaxArray | ComplexContinuation) -> ComplexContinuation:
        other_real, other_imag = complex_parts(other)
        return ComplexContinuation(
            self.real * other_real - self.imag * other_imag,
            self.real * other_imag + self.imag * other_real,
        )

    def __rmul__(self, other: JaxArray | ComplexContinuation) -> ComplexContinuation:
        return self * other

    def __truediv__(self, other: JaxArray | ComplexContinuation) -> ComplexContinuation:
        other_real, other_imag = complex_parts(other)
        denominator = other_real**2 + other_imag**2
        return ComplexContinuation(
            (self.real * other_real + self.imag * other_imag) / denominator,
            (self.imag * other_real - self.real * other_imag) / denominator,
        )


def analytic_continue_parameter(eta: JaxArray, delta: JaxArray) -> JaxArray | ComplexContinuation:
    """Continue real coordinates without treating complex storage as one coordinate."""
    xp = array_namespace(eta, delta)
    if xp.isdtype(eta.dtype, "complex floating"):
        return ComplexContinuation(
            xp.real(eta) + 1j * xp.real(delta),
            xp.imag(eta) + 1j * xp.imag(delta),
        )
    return eta + 1j * delta


def complex_parts(z: JaxArray | ComplexContinuation) -> tuple[JaxArray, JaxArray]:
    """Return the real-coordinate pair represented by a complex value."""
    if isinstance(z, ComplexContinuation):
        return z.real, z.imag
    xp = array_namespace(z)
    return xp.real(z), xp.imag(z)


def analytic_abs_square(z: JaxArray | ComplexContinuation) -> JaxArray:
    """Return ``real(z)**2 + imag(z)**2`` without conjugating continued coordinates."""
    real, imag = complex_parts(z)
    return real**2 + imag**2


def analytic_conj(z: JaxArray | ComplexContinuation) -> JaxArray | ComplexContinuation:
    """Return the coordinatewise analytic conjugate."""
    if isinstance(z, ComplexContinuation):
        return ComplexContinuation(z.real, -z.imag)
    xp = array_namespace(z)
    return xp.conj(z)


def analytic_real(z: JaxArray | ComplexContinuation) -> JaxArray:
    """Return the real coordinate of an analytic complex value."""
    if isinstance(z, ComplexContinuation):
        return z.real
    xp = array_namespace(z)
    return xp.real(z)


def complex_value(z: JaxArray | ComplexContinuation) -> JaxArray:
    """Return the ordinary complex value represented by real and imaginary coordinates."""
    real, imag = complex_parts(z)
    return real + 1j * imag
