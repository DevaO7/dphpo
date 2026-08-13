"""
Generic utilities for Rényi differential privacy accounting.

This module is mechanism-agnostic. It provides:

- a validated representation of an RDP curve;
- monotonicity post-processing for RDP upper bounds;
- pointwise scaling and composition of RDP curves;
- conversion from RDP to approximate differential privacy;
- small numerical helpers shared by privacy accountants.

Mechanism-specific accounting, such as DP-FedAvg or Papernot top-m
selection, should live in separate modules and communicate through
RdpCurve objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def gaussian_rdp(
    order: float,
    noise_multiplier: float,
) -> float:
    """Return the RDP epsilon of a Gaussian mechanism at one order.

    ``noise_multiplier`` is the Gaussian noise standard deviation divided
    by the mechanism's sensitivity.
    """
    order = float(order)
    noise_multiplier = float(noise_multiplier)

    if not math.isfinite(order) or order <= 1.0:
        raise ValueError("order must be finite and greater than 1.")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0.0:
        raise ValueError(
            "noise_multiplier must be finite and strictly positive."
        )

    return order / (2.0 * noise_multiplier**2)


def normalize_requested_integer_orders(
    orders: ArrayLike,
) -> NDArray[np.int64]:
    """Validate and return a strictly increasing integer Rényi-order grid.

    Accountants using this helper may evaluate every integer order from 2
    through the largest requested order internally, so sparse requested
    order sets are allowed.
    """
    orders_array = np.asarray(orders, dtype=float)

    if orders_array.ndim != 1:
        raise ValueError("orders must be one-dimensional.")
    if orders_array.size == 0:
        raise ValueError("orders must contain at least one order.")
    if not np.all(np.isfinite(orders_array)):
        raise ValueError("orders must contain only finite values.")
    if np.any(orders_array <= 1.0):
        raise ValueError(
            "Every requested Rényi order must be greater than 1."
        )
    if not np.all(
        np.isclose(
            orders_array,
            np.rint(orders_array),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "This accounting method supports integer Rényi orders only."
        )

    integer_orders = np.rint(orders_array).astype(np.int64)
    if np.any(np.diff(integer_orders) <= 0):
        raise ValueError(
            "Requested Rényi orders must be strictly increasing."
        )

    return integer_orders


def _as_readonly_float_array(
    values: ArrayLike,
    *,
    name: str,
) -> FloatArray:
    """
    Convert values to a one-dimensional, read-only float array.
    """
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    array = np.array(array, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _validate_delta(delta: float) -> float:
    delta = float(delta)

    if not math.isfinite(delta):
        raise ValueError("delta must be finite.")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must satisfy 0 < delta < 1.")

    return delta


def log1mexp(x: float) -> float:
    r"""
    Compute log(1 - exp(-x)) stably for x >= 0.

    This quantity appears in subsampling bounds. Directly evaluating

        log(1 - exp(-x))

    can lose precision when x is small.

    For x = 0, the mathematically correct result is -infinity.
    """
    x = float(x)

    if math.isnan(x):
        raise ValueError("x must not be NaN.")
    if x < 0.0:
        raise ValueError("x must satisfy x >= 0.")
    if x == 0.0:
        return -math.inf
    if math.isinf(x):
        return 0.0

    if x <= math.log(2.0):
        return math.log(-math.expm1(-x))

    return math.log1p(-math.exp(-x))


@dataclass(frozen=True, slots=True)
class RdpCurve:
    """
    RDP upper bounds evaluated at an increasing sequence of Rényi orders.

    Parameters
    ----------
    orders:
        Strictly increasing Rényi orders, each greater than 1.

    epsilons:
        Nonnegative RDP upper bounds corresponding to ``orders``.
    """

    orders: FloatArray
    epsilons: FloatArray

    def __post_init__(self) -> None:
        orders = _as_readonly_float_array(
            self.orders,
            name="orders",
        )
        epsilons = _as_readonly_float_array(
            self.epsilons,
            name="epsilons",
        )

        if orders.shape != epsilons.shape:
            raise ValueError(
                "orders and epsilons must have the same shape."
            )
        if np.any(orders <= 1.0):
            raise ValueError(
                "Every Rényi order must be strictly greater than 1."
            )
        if np.any(np.diff(orders) <= 0.0):
            raise ValueError(
                "Rényi orders must be strictly increasing."
            )
        if np.any(epsilons < 0.0):
            raise ValueError(
                "RDP epsilon values must be nonnegative."
            )

        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "epsilons", epsilons)

    @classmethod
    def from_mapping(
        cls,
        values_by_order: Mapping[float, float],
    ) -> "RdpCurve":
        """
        Construct an RDP curve from an order-to-epsilon mapping.

        The entries are sorted by Rényi order.
        """
        if not values_by_order:
            raise ValueError(
                "values_by_order must contain at least one entry."
            )

        sorted_items = sorted(
            (float(order), float(epsilon))
            for order, epsilon in values_by_order.items()
        )

        orders, epsilons = zip(*sorted_items)
        return cls(
            orders=np.asarray(orders, dtype=float),
            epsilons=np.asarray(epsilons, dtype=float),
        )

    def __len__(self) -> int:
        return int(self.orders.size)

    def as_dict(self) -> dict[float, float]:
        """
        Return a copy of the curve as an order-to-epsilon dictionary.
        """
        return {
            float(order): float(epsilon)
            for order, epsilon in zip(self.orders, self.epsilons)
        }

    def epsilon_at(
        self,
        order: float,
        *,
        atol: float = 1e-12,
    ) -> float:
        """
        Return the stored RDP epsilon at a specified order.

        This method does not interpolate. The requested order must already
        be present in the curve.
        """
        order = float(order)

        matches = np.flatnonzero(
            np.isclose(
                self.orders,
                order,
                rtol=0.0,
                atol=atol,
            )
        )

        if matches.size == 0:
            raise KeyError(
                f"Rényi order {order} is not present in the curve."
            )
        if matches.size > 1:
            raise RuntimeError(
                f"Multiple stored orders matched {order}."
            )

        return float(self.epsilons[int(matches[0])])


@dataclass(frozen=True, slots=True)
class ApproxDPResult:
    """
    Result of converting an RDP curve to approximate DP.
    """

    epsilon: float
    delta: float
    best_order: float
    best_index: int
    is_at_min_order: bool
    is_at_max_order: bool

    def __post_init__(self) -> None:
        epsilon = float(self.epsilon)
        delta = _validate_delta(self.delta)
        best_order = float(self.best_order)
        best_index = int(self.best_index)

        if not math.isfinite(epsilon):
            raise ValueError("epsilon must be finite.")
        if epsilon < 0.0:
            raise ValueError("epsilon must be nonnegative.")
        if not math.isfinite(best_order) or best_order <= 1.0:
            raise ValueError(
                "best_order must be finite and greater than 1."
            )
        if best_index < 0:
            raise ValueError("best_index must be nonnegative.")

        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "best_order", best_order)
        object.__setattr__(self, "best_index", best_index)


def apply_renyi_monotonicity_envelope(
    curve: RdpCurve,
) -> RdpCurve:
    r"""
    Improve an RDP upper-bound curve using monotonicity in the order.

    True RDP satisfies

        epsilon(alpha_1) <= epsilon(alpha_2)

    whenever alpha_1 < alpha_2. Therefore, if ``b_j`` is a valid upper
    bound at a larger order alpha_j, it is also a valid upper bound at
    every smaller stored order.

    The improved bound is

        improved[i] = min_{j >= i} curve.epsilons[j].
    """
    improved = np.minimum.accumulate(curve.epsilons[::-1])[::-1]

    return RdpCurve(
        orders=curve.orders,
        epsilons=improved,
    )


def scale_rdp_curve(
    curve: RdpCurve,
    factor: float,
) -> RdpCurve:
    """
    Multiply every RDP epsilon value by a nonnegative scalar.
    """
    factor = float(factor)

    if not math.isfinite(factor):
        raise ValueError("factor must be finite.")
    if factor < 0.0:
        raise ValueError("factor must be nonnegative.")

    return RdpCurve(
        orders=curve.orders,
        epsilons=factor * curve.epsilons,
    )


def compose_rdp_curves(
    *curves: RdpCurve,
) -> RdpCurve:
    """
    Compose RDP mechanisms by pointwise addition.

    All curves must be evaluated at exactly the same Rényi orders.
    """
    if not curves:
        raise ValueError(
            "At least one RDP curve is required for composition."
        )

    reference_orders = curves[0].orders
    total_epsilons = np.zeros_like(
        curves[0].epsilons,
        dtype=float,
    )

    for index, curve in enumerate(curves):
        if curve.orders.shape != reference_orders.shape or not np.array_equal(
            curve.orders,
            reference_orders,
        ):
            raise ValueError(
                "All RDP curves must use exactly the same Rényi orders. "
                f"Curve at position {index} does not match."
            )

        total_epsilons += curve.epsilons

    return RdpCurve(
        orders=reference_orders,
        epsilons=total_epsilons,
    )


def sum_rdp_curves(
    curves: Iterable[RdpCurve],
) -> RdpCurve:
    """
    Iterable-based wrapper around :func:`compose_rdp_curves`.
    """
    curves = tuple(curves)
    return compose_rdp_curves(*curves)


def convert_rdp_to_approx_dp(
    curve: RdpCurve,
    *,
    delta: float,
) -> ApproxDPResult:
    r"""
    Convert an RDP curve to an (epsilon, delta)-DP guarantee.

    This uses the standard conversion

        epsilon_DP(alpha)
        =
        epsilon_RDP(alpha)
        + log(1 / delta) / (alpha - 1),

    and minimizes the resulting bound over the stored Rényi orders.

    No interpolation or continuous-order optimization is performed.
    """
    delta = _validate_delta(delta)

    candidate_epsilons = (
        curve.epsilons
        + math.log(1.0 / delta) / (curve.orders - 1.0)
    )

    best_index = int(np.argmin(candidate_epsilons))

    return ApproxDPResult(
        epsilon=float(candidate_epsilons[best_index]),
        delta=delta,
        best_order=float(curve.orders[best_index]),
        best_index=best_index,
        is_at_min_order=(best_index == 0),
        is_at_max_order=(best_index == len(curve) - 1),
    )
