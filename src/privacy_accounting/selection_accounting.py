"""
Papernot-Steinke top-m and two-stage selection privacy accounting.

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
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rdp_utils import (
    RdpCurve,
    apply_renyi_monotonicity_envelope,
    compose_rdp_curves,
)
from tnb import TNBDistribution


FloatArray = NDArray[np.float64]


def _validate_positive_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if value < 1:
        raise ValueError(f"{name} must be at least 1.")

    return value


def _validate_eta(eta: float) -> float:
    eta = float(eta)

    if not math.isfinite(eta):
        raise ValueError("eta must be finite.")
    if eta <= -1.0:
        raise ValueError("eta must satisfy eta > -1.")

    return eta


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

    m = _validate_positive_integer(m, "m")
    eta = _validate_eta(eta)
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
    m = _validate_positive_integer(m, "m")

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
