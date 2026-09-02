"""
Papernot-Steinke selection privacy accounting.

This module operates only on precomputed RDP curves. It does not know how
the base mechanism was implemented: the curve may come from DP-FedAvg or
any other mechanism.

The top-m theorem implemented here is

    epsilon_top_m(lambda)
    =
    m * epsilon_base(lambda)
    + (m + eta) * (1 - 1 / lambda_hat)
        * epsilon_base(lambda_hat)
    + (m + eta) / lambda_hat * log(1 / gamma)
    + log E[binom(K, m)] / (lambda - 1),

where

    K = K0 | K0 >= m

and K0 follows the Papernot-Steinke truncated negative binomial
distribution.

Top-1 is implemented as the special case m = 1. A two-stage mechanism is
implemented by calling the same top-m accountant for stage 1, calling it
again with m = 1 for stage 2, and composing the two resulting RDP curves.

The unconditioned Poisson top-1 accountant is a separate implementation of
Theorem 6 from Papernot and Steinke (2022). The conditioned Poisson top-m
accountant implements the corresponding top-m theorem for
``K = K0 | K0 >= m``. A Poisson two-stage mechanism composes conditioned
top-m Stage 1 with unconditioned top-1 Stage 2, whose ``K0 = 0`` outcome is
handled by a public data-independent fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rdp_utils import (
    RdpCurve,
    apply_renyi_monotonicity_envelope,
    compose_rdp_curves,
)
from .poisson import PoissonDistribution
from .tnb import TNBDistribution
from .validation import validate_eta, validate_positive_integer


FloatArray = NDArray[np.float64]


def _validate_expected_num_trials(
    expected_num_trials: float,
    *,
    m: int,
) -> float:
    expected_num_trials = float(expected_num_trials)

    if not math.isfinite(expected_num_trials):
        raise ValueError(
            "expected_num_trials must be finite."
        )

    minimum = float(m)

    if expected_num_trials <= minimum:
        if m == 1:
            raise ValueError(
                "expected_num_trials must be greater than 1."
            )

        raise ValueError(
            "For top-m accounting, expected_num_trials represents "
            "E[K0 | K0 >= m] and must be strictly greater than "
            f"m={m}."
        )

    return expected_num_trials


def _build_tnb_distribution(
    *,
    eta: float,
    m: int,
    expected_num_trials: float,
) -> TNBDistribution:
    """
    Construct the TNB distribution matching the requested expected count.

    For m = 1, ``expected_num_trials`` means E[K0].

    For m > 1, ``expected_num_trials`` means

        E[K0 | K0 >= m].
    """
    if m == 1:
        return TNBDistribution.from_mean(
            eta=eta,
            target_mean=expected_num_trials,
        )

    return TNBDistribution.from_conditional_mean(
        eta=eta,
        m=m,
        target_mean=expected_num_trials,
    )


def _normalize_lambda_hat_orders(
    base_rdp_curve: RdpCurve,
    lambda_hat_orders: Optional[ArrayLike],
) -> FloatArray:
    """
    Return stored base-curve orders to use as lambda-hat candidates.

    The theorem candidate lambda_hat = 1 is always handled separately.
    Every supplied order must already be present in ``base_rdp_curve``;
    this module performs no interpolation.
    """
    if lambda_hat_orders is None:
        candidates = np.asarray(
            base_rdp_curve.orders,
            dtype=float,
        )
    else:
        candidates = np.asarray(
            lambda_hat_orders,
            dtype=float,
        )

        if candidates.ndim != 1:
            raise ValueError(
                "lambda_hat_orders must be one-dimensional."
            )
        if not np.all(np.isfinite(candidates)):
            raise ValueError(
                "lambda_hat_orders must contain only finite values."
            )

        # lambda_hat = 1 is already included explicitly.
        candidates = candidates[candidates > 1.0]

        candidates = np.unique(candidates)

        for order in candidates:
            try:
                base_rdp_curve.epsilon_at(float(order))
            except KeyError as exc:
                raise ValueError(
                    f"lambda_hat order {float(order)} is not present "
                    "in the base RDP curve."
                ) from exc

    if np.any(candidates <= 1.0):
        raise ValueError(
            "Every nontrivial lambda_hat candidate must be greater "
            "than 1."
        )

    return np.array(
        candidates,
        dtype=float,
        copy=True,
    )


@dataclass(frozen=True, slots=True)
class HatTermResult:
    """
    Result of optimizing the auxiliary lambda-hat theorem term.
    """

    value: float
    best_lambda_hat: float

    def __post_init__(self) -> None:
        value = float(self.value)
        best_lambda_hat = float(self.best_lambda_hat)

        if not math.isfinite(value):
            raise ValueError(
                "The optimized lambda-hat term must be finite."
            )
        if value < 0.0:
            raise ValueError(
                "The optimized lambda-hat term must be nonnegative."
            )
        if (
            not math.isfinite(best_lambda_hat)
            or best_lambda_hat < 1.0
        ):
            raise ValueError(
                "best_lambda_hat must be finite and at least 1."
            )

        object.__setattr__(self, "value", value)
        object.__setattr__(
            self,
            "best_lambda_hat",
            best_lambda_hat,
        )


@dataclass(frozen=True, slots=True)
class TopMResult:
    """
    Privacy-accounting result for direct top-m release.

    Attributes
    ----------
    rdp_curve:
        The complete top-m RDP upper-bound curve.

    base_rdp_curve:
        The base mechanism curve used to construct the result.

    distribution:
        The TNB distribution K0. For m > 1, the theorem uses
        K = K0 | K0 >= m.

    expected_num_trials:
        E[K0] for m = 1, or E[K0 | K0 >= m] for m > 1.

    log_expected_binomial:
        log E[binom(K, m)] for the theorem's repetition variable K.

    best_lambda_hat:
        Best auxiliary order among lambda_hat = 1 and the requested
        stored base-curve orders.
    """

    rdp_curve: RdpCurve
    base_rdp_curve: RdpCurve
    distribution: TNBDistribution
    m: int
    eta: float
    expected_num_trials: float
    log_expected_binomial: float
    best_hat_term: float
    best_lambda_hat: float
    monotonicity_applied: bool

    @property
    def gamma(self) -> float:
        return self.distribution.gamma


@dataclass(frozen=True, slots=True)
class TwoStageResult:
    """
    RDP-accounting result for a two-stage selection mechanism.

    Stage 1 releases top-m.
    Stage 2 releases top-1.
    """

    rdp_curve: RdpCurve
    stage_1: TopMResult
    stage_2: TopMResult
    monotonicity_applied: bool


def _readonly_float_array(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array = np.array(array, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class PoissonTop1Result:
    """Theorem-6 Poisson top-1 curve and conversion diagnostics."""

    rdp_curve: RdpCurve
    raw_rdp_curve: RdpCurve
    base_rdp_curve: RdpCurve
    distribution: PoissonDistribution
    hat_epsilons: FloatArray
    hat_deltas: FloatArray
    best_auxiliary_orders: FloatArray
    envelope_source_orders: FloatArray
    monotonicity_applied: bool

    def __post_init__(self) -> None:
        if not isinstance(self.base_rdp_curve, RdpCurve):
            raise TypeError("base_rdp_curve must be an RdpCurve.")
        if not isinstance(self.distribution, PoissonDistribution):
            raise TypeError("distribution must be a PoissonDistribution.")
        for curve_name in ("raw_rdp_curve", "rdp_curve"):
            curve = getattr(self, curve_name)
            if not isinstance(curve, RdpCurve):
                raise TypeError(f"{curve_name} must be an RdpCurve.")
            if not np.array_equal(
                curve.orders,
                self.base_rdp_curve.orders,
            ):
                raise ValueError(
                    f"{curve_name} must use the base RDP order grid."
                )
        expected_shape = self.base_rdp_curve.orders.shape
        for field_name in (
            "hat_epsilons",
            "hat_deltas",
            "best_auxiliary_orders",
            "envelope_source_orders",
        ):
            array = _readonly_float_array(
                getattr(self, field_name),
                field_name,
            )
            if array.shape != expected_shape:
                raise ValueError(
                    f"{field_name} must match the base RDP order grid."
                )
            object.__setattr__(self, field_name, array)
        if np.any(self.hat_epsilons < 0.0):
            raise ValueError("hat_epsilons must be nonnegative.")
        if np.any((self.hat_deltas < 0.0) | (self.hat_deltas > 1.0)):
            raise ValueError("hat_deltas must lie in [0, 1].")
        if np.any(self.best_auxiliary_orders <= 1.0):
            raise ValueError("best_auxiliary_orders must exceed 1.")
        if np.any(self.envelope_source_orders < self.rdp_curve.orders):
            raise ValueError(
                "An RDP envelope source order cannot be below its target order."
            )

    @property
    def mu(self) -> float:
        return self.distribution.mu


@dataclass(frozen=True, slots=True)
class PoissonTopMResult:
    """Conditioned-Poisson top-m curve and conversion diagnostics."""

    rdp_curve: RdpCurve
    raw_rdp_curve: RdpCurve
    base_rdp_curve: RdpCurve
    distribution: PoissonDistribution
    m: int
    expected_num_trials: float
    log_expected_binomial: float
    hat_epsilons: FloatArray
    hat_deltas: FloatArray
    best_auxiliary_orders: FloatArray
    envelope_source_orders: FloatArray
    monotonicity_applied: bool

    def __post_init__(self) -> None:
        if not isinstance(self.base_rdp_curve, RdpCurve):
            raise TypeError("base_rdp_curve must be an RdpCurve.")
        if not isinstance(self.distribution, PoissonDistribution):
            raise TypeError("distribution must be a PoissonDistribution.")
        object.__setattr__(
            self,
            "m",
            validate_positive_integer(self.m, "m"),
        )
        expected_num_trials = float(self.expected_num_trials)
        if (
            not math.isfinite(expected_num_trials)
            or expected_num_trials <= self.m
        ):
            raise ValueError(
                "expected_num_trials must be finite and strictly greater "
                "than m for conditioned-Poisson top-m accounting."
            )
        object.__setattr__(
            self,
            "expected_num_trials",
            expected_num_trials,
        )
        log_expected_binomial = float(self.log_expected_binomial)
        if (
            not math.isfinite(log_expected_binomial)
            or log_expected_binomial < 0.0
        ):
            raise ValueError(
                "log_expected_binomial must be finite and nonnegative."
            )
        object.__setattr__(
            self,
            "log_expected_binomial",
            log_expected_binomial,
        )
        for curve_name in ("raw_rdp_curve", "rdp_curve"):
            curve = getattr(self, curve_name)
            if not isinstance(curve, RdpCurve):
                raise TypeError(f"{curve_name} must be an RdpCurve.")
            if not np.array_equal(
                curve.orders,
                self.base_rdp_curve.orders,
            ):
                raise ValueError(
                    f"{curve_name} must use the base RDP order grid."
                )
        expected_shape = self.base_rdp_curve.orders.shape
        for field_name in (
            "hat_epsilons",
            "hat_deltas",
            "best_auxiliary_orders",
            "envelope_source_orders",
        ):
            array = _readonly_float_array(
                getattr(self, field_name),
                field_name,
            )
            if array.shape != expected_shape:
                raise ValueError(
                    f"{field_name} must match the base RDP order grid."
                )
            object.__setattr__(self, field_name, array)
        if np.any(self.hat_epsilons < 0.0):
            raise ValueError("hat_epsilons must be nonnegative.")
        if np.any((self.hat_deltas < 0.0) | (self.hat_deltas > 1.0)):
            raise ValueError("hat_deltas must lie in [0, 1].")
        if np.any(self.best_auxiliary_orders <= 1.0):
            raise ValueError("best_auxiliary_orders must exceed 1.")
        if np.any(self.envelope_source_orders < self.rdp_curve.orders):
            raise ValueError(
                "An RDP envelope source order cannot be below its target "
                "order."
            )

    @property
    def mu(self) -> float:
        """Return the underlying, unconditioned Poisson rate."""
        return self.distribution.mu


@dataclass(frozen=True, slots=True)
class PoissonTwoStageResult:
    """RDP result for conditioned top-m then unconditioned top-1."""

    rdp_curve: RdpCurve
    stage_1: PoissonTopMResult
    stage_2: PoissonTop1Result
    monotonicity_applied: bool

    def __post_init__(self) -> None:
        if not isinstance(self.rdp_curve, RdpCurve):
            raise TypeError("rdp_curve must be an RdpCurve.")
        if not isinstance(self.stage_1, PoissonTopMResult):
            raise TypeError("stage_1 must be a PoissonTopMResult.")
        if not isinstance(self.stage_2, PoissonTop1Result):
            raise TypeError("stage_2 must be a PoissonTop1Result.")
        if not np.array_equal(
            self.rdp_curve.orders,
            self.stage_1.rdp_curve.orders,
        ):
            raise ValueError(
                "The composed curve must use the Stage-1 RDP order grid."
            )


@dataclass(frozen=True, slots=True)
class PoissonNStageResult:
    """RDP-accounting result for a fixed multi-stage Poisson mechanism.

    Every stage except the last releases an ordered top-``m`` output from
    a Poisson count conditioned to be at least ``m``.  The last stage uses
    the unconditioned Papernot--Steinke Poisson top-1 mechanism, including
    its data-independent ``K = 0`` fallback.  The stage guarantees are
    composed pointwise on one common Renyi-order grid.
    """

    rdp_curve: RdpCurve
    stage_results: tuple[PoissonTopMResult | PoissonTop1Result, ...]
    monotonicity_applied: bool

    def __post_init__(self) -> None:
        if not isinstance(self.rdp_curve, RdpCurve):
            raise TypeError("rdp_curve must be an RdpCurve.")
        stage_results = tuple(self.stage_results)
        if not stage_results:
            raise ValueError("stage_results must contain at least one stage.")
        for stage_index, stage_result in enumerate(stage_results, start=1):
            if not isinstance(
                stage_result,
                (PoissonTopMResult, PoissonTop1Result),
            ):
                raise TypeError(
                    "Every stage result must be PoissonTopMResult or "
                    f"PoissonTop1Result; stage {stage_index} has "
                    f"{type(stage_result).__name__}."
                )
            if not np.array_equal(
                stage_result.rdp_curve.orders,
                self.rdp_curve.orders,
            ):
                raise ValueError(
                    f"Stage {stage_index} does not use the composed "
                    "Renyi-order grid."
                )
            if (
                stage_index < len(stage_results)
                and not isinstance(stage_result, PoissonTopMResult)
            ):
                raise ValueError(
                    "Every non-final stage must use conditioned-Poisson "
                    "top-m accounting."
                )
        if not isinstance(stage_results[-1], PoissonTop1Result):
            raise ValueError(
                "The final stage must use the unconditioned Poisson "
                "top-1 mechanism."
            )
        object.__setattr__(self, "stage_results", stage_results)


def _normalize_auxiliary_orders(
    base_rdp_curve: RdpCurve,
    auxiliary_orders: Optional[ArrayLike],
) -> FloatArray:
    if auxiliary_orders is None:
        return np.array(base_rdp_curve.orders, dtype=float, copy=True)
    orders = np.asarray(auxiliary_orders, dtype=float)
    if orders.ndim != 1 or orders.size == 0:
        raise ValueError(
            "auxiliary_orders must be a non-empty one-dimensional array."
        )
    if not np.all(np.isfinite(orders)) or np.any(orders <= 1.0):
        raise ValueError(
            "Every auxiliary order must be finite and greater than 1."
        )
    orders = np.unique(orders)
    for order in orders:
        try:
            base_rdp_curve.epsilon_at(float(order))
        except KeyError as error:
            raise ValueError(
                f"Auxiliary order {float(order)} is not stored in the "
                "base RDP curve."
            ) from error
    return np.array(orders, dtype=float, copy=True)


def _renyi_envelope_source_indices(epsilons: FloatArray) -> NDArray[np.int64]:
    """Return the raw higher-order bound used at every target order."""
    source_indices = np.empty(epsilons.size, dtype=np.int64)
    best_index = epsilons.size - 1
    for index in range(epsilons.size - 1, -1, -1):
        if epsilons[index] <= epsilons[best_index]:
            best_index = index
        source_indices[index] = best_index
    return source_indices


def _compute_poisson_hat_delta_diagnostics(
    *,
    base_rdp_curve: RdpCurve,
    target_orders: FloatArray,
    auxiliary_orders: Optional[ArrayLike],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Optimize the common RDP-to-approximate-DP term over orders."""
    normalized_auxiliary_orders = _normalize_auxiliary_orders(
        base_rdp_curve,
        auxiliary_orders,
    )
    auxiliary_epsilons = np.asarray(
        [
            base_rdp_curve.epsilon_at(float(order))
            for order in normalized_auxiliary_orders
        ],
        dtype=float,
    )

    # Equality is optimal in the theorem condition: increasing
    # hat_epsilon decreases the converted hat_delta.
    hat_epsilons = np.log1p(1.0 / (target_orders - 1.0))
    alpha_minus_one = normalized_auxiliary_orders - 1.0
    log_conversion_constants = (
        -np.log(normalized_auxiliary_orders)
        + alpha_minus_one
        * np.log1p(-1.0 / normalized_auxiliary_orders)
    )
    log_hat_delta_candidates = (
        log_conversion_constants[:, None]
        + alpha_minus_one[:, None]
        * (
            auxiliary_epsilons[:, None]
            - hat_epsilons[None, :]
        )
    )
    best_auxiliary_indices = np.argmin(
        log_hat_delta_candidates,
        axis=0,
    )
    best_log_hat_deltas = log_hat_delta_candidates[
        best_auxiliary_indices,
        np.arange(target_orders.size),
    ]
    hat_deltas = np.exp(np.minimum(best_log_hat_deltas, 0.0))
    best_auxiliary_orders = normalized_auxiliary_orders[
        best_auxiliary_indices
    ]
    return (
        np.asarray(hat_epsilons, dtype=float),
        np.asarray(hat_deltas, dtype=float),
        np.asarray(best_auxiliary_orders, dtype=float),
    )


def compute_top1_rdp_poisson(
    base_rdp_curve: RdpCurve,
    *,
    expected_num_trials: float,
    auxiliary_orders: Optional[ArrayLike] = None,
    apply_monotonicity: bool = True,
) -> PoissonTop1Result:
    r"""Compute Papernot--Steinke Theorem 6 for ``K ~ Poisson(mu)``.

    At output order :math:`\lambda`, Theorem 6 gives

    .. math::

        \varepsilon_A(\lambda)
        = \varepsilon_Q(\lambda) + \mu\hat\delta
          + \frac{\log\mu}{\lambda-1},

    provided ``Q`` is also ``(hat_epsilon, hat_delta)``-DP and
    ``exp(hat_epsilon) <= 1 + 1 / (lambda - 1)``.  We use the largest
    admissible ``hat_epsilon`` and derive ``hat_delta`` from every stored
    auxiliary RDP order ``alpha`` using the tight RDP-to-DP conversion

    .. math::

        \hat\delta = \frac1\alpha
        (1-1/\alpha)^{\alpha-1}
        \exp((\alpha-1)(\varepsilon_Q(\alpha)-\hat\varepsilon)).

    The sign in the exponent is therefore base RDP epsilon minus the
    requested approximate-DP epsilon.  A value above one is replaced by
    the valid trivial guarantee ``hat_delta = 1``.
    """
    if not isinstance(base_rdp_curve, RdpCurve):
        raise TypeError("base_rdp_curve must be an RdpCurve.")
    distribution = PoissonDistribution(expected_num_trials)
    target_orders = np.asarray(base_rdp_curve.orders, dtype=float)
    (
        hat_epsilons,
        hat_deltas,
        best_auxiliary_orders,
    ) = _compute_poisson_hat_delta_diagnostics(
        base_rdp_curve=base_rdp_curve,
        target_orders=target_orders,
        auxiliary_orders=auxiliary_orders,
    )

    raw_epsilons = (
        base_rdp_curve.epsilons
        + distribution.mu * hat_deltas
        + math.log(distribution.mu) / (target_orders - 1.0)
    )
    # RDP is nonnegative. This also removes negligible negative roundoff in
    # otherwise valid theorem bounds for very small means.
    raw_epsilons = np.maximum(raw_epsilons, 0.0)
    raw_curve = RdpCurve(
        orders=target_orders,
        epsilons=raw_epsilons,
    )

    if apply_monotonicity:
        source_indices = _renyi_envelope_source_indices(raw_epsilons)
        rdp_curve = apply_renyi_monotonicity_envelope(raw_curve)
    else:
        source_indices = np.arange(target_orders.size, dtype=np.int64)
        rdp_curve = raw_curve

    return PoissonTop1Result(
        rdp_curve=rdp_curve,
        raw_rdp_curve=raw_curve,
        base_rdp_curve=base_rdp_curve,
        distribution=distribution,
        hat_epsilons=hat_epsilons,
        hat_deltas=hat_deltas,
        best_auxiliary_orders=best_auxiliary_orders,
        envelope_source_orders=target_orders[source_indices],
        monotonicity_applied=bool(apply_monotonicity),
    )


def compute_top_m_rdp_poisson(
    base_rdp_curve: RdpCurve,
    *,
    m: int,
    expected_num_trials: float,
    auxiliary_orders: Optional[ArrayLike] = None,
    apply_monotonicity: bool = True,
) -> PoissonTopMResult:
    r"""Compute conditioned-Poisson top-m RDP accounting.

    ``expected_num_trials`` is the conditional expected count

    .. math::

        \mathbb E[K_0 \mid K_0 \ge m].

    The underlying Poisson rate ``mu`` is calibrated to this target. At
    output order :math:`\lambda`, the theorem gives

    .. math::

        \varepsilon_A(\lambda)
        = m\varepsilon_Q(\lambda)
          + \mu\hat\delta
          + \frac{\log\mathbb E[\binom K m]}{\lambda-1},

    where ``K = K0 | K0 >= m``. The approximate-DP conversion and
    auxiliary-order optimization are exactly the same as in the
    unconditioned Papernot--Steinke Poisson top-1 accountant.
    """
    if not isinstance(base_rdp_curve, RdpCurve):
        raise TypeError("base_rdp_curve must be an RdpCurve.")
    m = validate_positive_integer(m, "m")
    expected_num_trials = _validate_expected_num_trials(
        expected_num_trials,
        m=m,
    )
    distribution = PoissonDistribution.from_conditional_mean(
        m=m,
        target_mean=expected_num_trials,
    )
    target_orders = np.asarray(base_rdp_curve.orders, dtype=float)
    (
        hat_epsilons,
        hat_deltas,
        best_auxiliary_orders,
    ) = _compute_poisson_hat_delta_diagnostics(
        base_rdp_curve=base_rdp_curve,
        target_orders=target_orders,
        auxiliary_orders=auxiliary_orders,
    )
    log_expected_binomial = distribution.log_expected_binomial(m)
    raw_epsilons = (
        m * base_rdp_curve.epsilons
        + distribution.mu * hat_deltas
        + log_expected_binomial / (target_orders - 1.0)
    )
    raw_curve = RdpCurve(
        orders=target_orders,
        epsilons=np.maximum(raw_epsilons, 0.0),
    )
    if apply_monotonicity:
        source_indices = _renyi_envelope_source_indices(raw_epsilons)
        rdp_curve = apply_renyi_monotonicity_envelope(raw_curve)
    else:
        source_indices = np.arange(target_orders.size, dtype=np.int64)
        rdp_curve = raw_curve

    return PoissonTopMResult(
        rdp_curve=rdp_curve,
        raw_rdp_curve=raw_curve,
        base_rdp_curve=base_rdp_curve,
        distribution=distribution,
        m=m,
        expected_num_trials=expected_num_trials,
        log_expected_binomial=log_expected_binomial,
        hat_epsilons=hat_epsilons,
        hat_deltas=hat_deltas,
        best_auxiliary_orders=best_auxiliary_orders,
        envelope_source_orders=target_orders[source_indices],
        monotonicity_applied=bool(apply_monotonicity),
    )


def compute_two_stage_rdp_poisson(
    stage_1_base_rdp_curve: RdpCurve,
    stage_2_base_rdp_curve: RdpCurve,
    *,
    m: int,
    expected_num_trials_stage_1: float,
    expected_num_trials_stage_2: Optional[float] = None,
    auxiliary_orders_stage_1: Optional[ArrayLike] = None,
    auxiliary_orders_stage_2: Optional[ArrayLike] = None,
    apply_monotonicity: bool = True,
) -> PoissonTwoStageResult:
    r"""Compose conditioned top-m Stage 1 and unconditioned top-1 Stage 2.

    Stage 1 always samples at least ``m`` runs. Stage 2 uses
    ``K2 ~ Poisson(mu2)`` without conditioning; when ``K2 = 0`` it returns
    the mechanism's predefined data-independent fallback. Thus Stage 2 is
    accounted using Papernot--Steinke Theorem 6.
    """
    m = validate_positive_integer(m, "m")
    if m <= 1:
        raise ValueError(
            "A nondegenerate two-stage mechanism requires m > 1."
        )
    if expected_num_trials_stage_2 is None:
        expected_num_trials_stage_2 = float(m)

    stage_1_result = compute_top_m_rdp_poisson(
        stage_1_base_rdp_curve,
        m=m,
        expected_num_trials=expected_num_trials_stage_1,
        auxiliary_orders=auxiliary_orders_stage_1,
        apply_monotonicity=apply_monotonicity,
    )
    stage_2_result = compute_top1_rdp_poisson(
        stage_2_base_rdp_curve,
        expected_num_trials=expected_num_trials_stage_2,
        auxiliary_orders=auxiliary_orders_stage_2,
        apply_monotonicity=apply_monotonicity,
    )
    composed_curve = compose_rdp_curves(
        stage_1_result.rdp_curve,
        stage_2_result.rdp_curve,
    )
    if apply_monotonicity:
        composed_curve = apply_renyi_monotonicity_envelope(
            composed_curve
        )

    return PoissonTwoStageResult(
        rdp_curve=composed_curve,
        stage_1=stage_1_result,
        stage_2=stage_2_result,
        monotonicity_applied=bool(apply_monotonicity),
    )


def compute_n_stage_rdp_poisson(
    base_rdp_curves: Sequence[RdpCurve],
    *,
    retained_counts: Sequence[int],
    expected_num_trials: Sequence[float],
    auxiliary_orders_by_stage: Optional[
        Sequence[Optional[ArrayLike]]
    ] = None,
    apply_monotonicity: bool = True,
) -> PoissonNStageResult:
    r"""Compose a fixed multi-stage Poisson selection schedule.

    For stages ``1, ..., L - 1``, ``expected_num_trials[l]`` denotes the
    conditional expectation

    .. math::

        \bar\mu_l = E[K_{0,l} \mid K_{0,l} \ge m_l],

    and the underlying Poisson rate is calibrated internally.  The final
    stage must have ``retained_counts[-1] == 1`` and uses unconditioned
    Poisson top-1 accounting, so its expected count is its Poisson rate and
    ``K_L = 0`` returns the predefined data-independent fallback.

    The schedule is fixed before the mechanism runs.  All stage guarantees
    are added at the same Renyi order before applying the monotonicity
    envelope and converting to approximate DP.
    """
    base_rdp_curves = tuple(base_rdp_curves)
    retained_counts = tuple(retained_counts)
    expected_num_trials = tuple(expected_num_trials)
    num_stages = len(base_rdp_curves)
    if num_stages == 0:
        raise ValueError("base_rdp_curves must contain at least one stage.")
    if len(retained_counts) != num_stages:
        raise ValueError(
            "retained_counts must contain one entry per base RDP curve."
        )
    if len(expected_num_trials) != num_stages:
        raise ValueError(
            "expected_num_trials must contain one entry per base RDP "
            "curve."
        )
    if auxiliary_orders_by_stage is None:
        auxiliary_orders = (None,) * num_stages
    else:
        auxiliary_orders = tuple(auxiliary_orders_by_stage)
        if len(auxiliary_orders) != num_stages:
            raise ValueError(
                "auxiliary_orders_by_stage must contain one entry per "
                "base RDP curve."
            )

    normalized_retained_counts = tuple(
        validate_positive_integer(count, f"retained_counts[{index}]")
        for index, count in enumerate(retained_counts)
    )
    if normalized_retained_counts[-1] != 1:
        raise ValueError(
            "The final stage must retain one output and use unconditioned "
            "Poisson top-1 accounting."
        )

    stage_results: list[PoissonTopMResult | PoissonTop1Result] = []
    for stage_index, (
        base_rdp_curve,
        retained_count,
        stage_expected_num_trials,
        stage_auxiliary_orders,
    ) in enumerate(
        zip(
            base_rdp_curves,
            normalized_retained_counts,
            expected_num_trials,
            auxiliary_orders,
        )
    ):
        if stage_index == num_stages - 1:
            stage_result = compute_top1_rdp_poisson(
                base_rdp_curve,
                expected_num_trials=stage_expected_num_trials,
                auxiliary_orders=stage_auxiliary_orders,
                apply_monotonicity=apply_monotonicity,
            )
        else:
            stage_result = compute_top_m_rdp_poisson(
                base_rdp_curve,
                m=retained_count,
                expected_num_trials=stage_expected_num_trials,
                auxiliary_orders=stage_auxiliary_orders,
                apply_monotonicity=apply_monotonicity,
            )
        stage_results.append(stage_result)

    composed_curve = compose_rdp_curves(
        *(stage_result.rdp_curve for stage_result in stage_results)
    )
    if apply_monotonicity:
        composed_curve = apply_renyi_monotonicity_envelope(composed_curve)

    return PoissonNStageResult(
        rdp_curve=composed_curve,
        stage_results=tuple(stage_results),
        monotonicity_applied=bool(apply_monotonicity),
    )


def _find_best_hat_term(
    *,
    m: int,
    eta: float,
    gamma: float,
    base_rdp_curve: RdpCurve,
    lambda_hat_orders: Optional[ArrayLike] = None,
) -> HatTermResult:
    r"""
    Minimize the lambda-hat-dependent theorem term.

    The candidate lambda_hat = 1 contributes

        (m + eta) log(1 / gamma),

    because the epsilon(lambda_hat) term vanishes.
    """
    coefficient = m + eta
    log_inverse_gamma = math.log(1.0 / gamma)

    best_value = coefficient * log_inverse_gamma
    best_lambda_hat = 1.0

    candidates = _normalize_lambda_hat_orders(
        base_rdp_curve,
        lambda_hat_orders,
    )

    for lambda_hat in candidates:
        epsilon_hat = base_rdp_curve.epsilon_at(
            float(lambda_hat)
        )

        value = coefficient * (
            (1.0 - 1.0 / lambda_hat) * epsilon_hat
            + log_inverse_gamma / lambda_hat
        )

        if value < best_value:
            best_value = float(value)
            best_lambda_hat = float(lambda_hat)

    return HatTermResult(
        value=best_value,
        best_lambda_hat=best_lambda_hat,
    )


def compute_top_m_rdp(
    base_rdp_curve: RdpCurve,
    *,
    m: int,
    expected_num_trials: float,
    eta: float = 0.0,
    lambda_hat_orders: Optional[ArrayLike] = None,
    apply_monotonicity: bool = True,
) -> TopMResult:
    r"""
    Compute the Papernot-Steinke top-m RDP curve.

    Parameters
    ----------
    base_rdp_curve:
        RDP curve of one execution of the base mechanism.

    m:
        Number of HP-model pairs released.

    expected_num_trials:
        For ``m = 1``, this is E[K0].

        For ``m > 1``, this is

            E[K0 | K0 >= m].

    eta:
        TNB shape parameter satisfying eta > -1.

    lambda_hat_orders:
        Optional stored base-curve orders over which to optimize
        lambda_hat. The candidate lambda_hat = 1 is always included.
        If omitted, all stored base-curve orders are used.

    apply_monotonicity:
        Whether to improve the final top-m RDP curve using Rényi-order
        monotonicity.

    Returns
    -------
    TopMResult
        The top-m RDP curve and theorem diagnostics.
    """
    if not isinstance(base_rdp_curve, RdpCurve):
        raise TypeError(
            "base_rdp_curve must be an RdpCurve."
        )

    m = validate_positive_integer(m, "m")
    eta = validate_eta(eta)
    expected_num_trials = _validate_expected_num_trials(
        expected_num_trials,
        m=m,
    )

    distribution = _build_tnb_distribution(
        eta=eta,
        m=m,
        expected_num_trials=expected_num_trials,
    )

    hat_result = _find_best_hat_term(
        m=m,
        eta=eta,
        gamma=distribution.gamma,
        base_rdp_curve=base_rdp_curve,
        lambda_hat_orders=lambda_hat_orders,
    )

    log_expected_binomial = (
        distribution.log_expected_binomial(m)
    )

    theorem_epsilons = (
        m * base_rdp_curve.epsilons
        + hat_result.value
        + log_expected_binomial
        / (base_rdp_curve.orders - 1.0)
    )

    top_m_curve = RdpCurve(
        orders=base_rdp_curve.orders,
        epsilons=theorem_epsilons,
    )

    if apply_monotonicity:
        top_m_curve = apply_renyi_monotonicity_envelope(
            top_m_curve
        )

    return TopMResult(
        rdp_curve=top_m_curve,
        base_rdp_curve=base_rdp_curve,
        distribution=distribution,
        m=m,
        eta=eta,
        expected_num_trials=expected_num_trials,
        log_expected_binomial=float(
            log_expected_binomial
        ),
        best_hat_term=hat_result.value,
        best_lambda_hat=hat_result.best_lambda_hat,
        monotonicity_applied=bool(apply_monotonicity),
    )


def compute_top1_rdp(
    base_rdp_curve: RdpCurve,
    *,
    expected_num_trials: float,
    eta: float = 0.0,
    lambda_hat_orders: Optional[ArrayLike] = None,
    apply_monotonicity: bool = True,
) -> TopMResult:
    """
    Convenience wrapper for direct top-1 release.

    This function delegates entirely to :func:`compute_top_m_rdp` with
    ``m=1``; it does not maintain a separate theorem implementation.
    """
    return compute_top_m_rdp(
        base_rdp_curve,
        m=1,
        expected_num_trials=expected_num_trials,
        eta=eta,
        lambda_hat_orders=lambda_hat_orders,
        apply_monotonicity=apply_monotonicity,
    )


def compute_two_stage_rdp(
    stage_1_base_rdp_curve: RdpCurve,
    stage_2_base_rdp_curve: RdpCurve,
    *,
    m: int,
    expected_num_trials_stage_1: float,
    expected_num_trials_stage_2: Optional[float] = None,
    eta_stage_1: float = 0.0,
    eta_stage_2: Optional[float] = None,
    lambda_hat_orders_stage_1: Optional[ArrayLike] = None,
    lambda_hat_orders_stage_2: Optional[ArrayLike] = None,
    apply_monotonicity: bool = True,
) -> TwoStageResult:
    r"""
    Compute the RDP curve of a two-stage selection mechanism.

    Stage 1
        Apply the top-m theorem to ``stage_1_base_rdp_curve``.

    Stage 2
        Apply the top-1 theorem to ``stage_2_base_rdp_curve``.

    The two stage curves are then composed pointwise at the same Rényi
    orders.

    Parameters
    ----------
    stage_1_base_rdp_curve:
        Base mechanism used in the lower-resource top-m stage.

    stage_2_base_rdp_curve:
        Base mechanism used in the higher-resource top-1 stage.

    m:
        Number of candidates retained/released by stage 1. Must be
        greater than 1 for a nondegenerate two-stage mechanism.

    expected_num_trials_stage_1:
        E[K0 | K0 >= m] for the stage-1 repetition distribution.

    expected_num_trials_stage_2:
        E[K0] for the stage-2 top-1 repetition distribution. If omitted,
        defaults to ``m``.

    eta_stage_1, eta_stage_2:
        TNB shape parameters for the two stages. If ``eta_stage_2`` is
        omitted, it defaults to ``eta_stage_1``.

    lambda_hat_orders_stage_1, lambda_hat_orders_stage_2:
        Optional stored orders used to optimize the two auxiliary
        lambda-hat terms.

    apply_monotonicity:
        Whether to improve each stage curve and the composed curve using
        Rényi-order monotonicity.
    """
    m = validate_positive_integer(m, "m")

    if m <= 1:
        raise ValueError(
            "A nondegenerate two-stage mechanism requires m > 1."
        )

    if expected_num_trials_stage_2 is None:
        expected_num_trials_stage_2 = float(m)

    if eta_stage_2 is None:
        eta_stage_2 = eta_stage_1

    stage_1_result = compute_top_m_rdp(
        stage_1_base_rdp_curve,
        m=m,
        expected_num_trials=expected_num_trials_stage_1,
        eta=eta_stage_1,
        lambda_hat_orders=lambda_hat_orders_stage_1,
        apply_monotonicity=apply_monotonicity,
    )

    stage_2_result = compute_top1_rdp(
        stage_2_base_rdp_curve,
        expected_num_trials=expected_num_trials_stage_2,
        eta=eta_stage_2,
        lambda_hat_orders=lambda_hat_orders_stage_2,
        apply_monotonicity=apply_monotonicity,
    )

    composed_curve = compose_rdp_curves(
        stage_1_result.rdp_curve,
        stage_2_result.rdp_curve,
    )

    if apply_monotonicity:
        composed_curve = apply_renyi_monotonicity_envelope(
            composed_curve
        )

    return TwoStageResult(
        rdp_curve=composed_curve,
        stage_1=stage_1_result,
        stage_2=stage_2_result,
        monotonicity_applied=bool(apply_monotonicity),
    )
