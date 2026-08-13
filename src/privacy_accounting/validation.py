"""Shared input validation for privacy-accounting mechanisms."""

import math
from numbers import Integral


def validate_positive_integer(value: int, name: str) -> int:
    """Return ``value`` as an int after requiring an integer of at least 1."""
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")

    return value


def validate_probability(value: float, name: str) -> float:
    """Return a finite probability in the interval ``(0, 1]``."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must satisfy 0 < {name} <= 1.")

    return value


def validate_positive_float(value: float, name: str) -> float:
    """Return a finite, strictly positive float."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")

    return value


def validate_eta(eta: float) -> float:
    """Validate the TNB shape parameter shared by selection accountants."""
    eta = float(eta)
    if not math.isfinite(eta):
        raise ValueError("eta must be finite.")
    if eta <= -1.0:
        raise ValueError("eta must satisfy eta > -1.")

    return eta
