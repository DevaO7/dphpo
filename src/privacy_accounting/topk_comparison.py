"""Pure calculations for the exp7 single-stage top-k comparison.

TCT evaluates each of N configurations once. Our selection mechanism uses
K ~ Poisson(mu) | K >= k and samples configurations with replacement.
Coverage-constrained comparisons impose a minimum coverage requirement.
Expected-compute comparisons instead calibrate the conditional mean to N.
Compute is measured in optimizer updates: N*T versus E[K]*T. This module
does not run training, sample K, save files, or plot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.optimize import brentq

from .dp_composition import compose_approx_dp_from_rdp
from .poisson import PoissonDistribution
from .rdp_utils import (
    ApproxDPResult,
    RdpCurve,
    convert_rdp_to_approx_dp_with_policy,
    scale_rdp_curve,
)
from .schedule_optimization import conditioned_all_configurations_coverage_probability
from .selection_accounting import compute_top_m_rdp_poisson
from .tct import (
    TCTInfeasibleError,
    TCTOptimizerConfig,
    open_probability,
    positive_integer,
)
from .tct_rdp import tct_topk_from_rdp


class CoverageCalibrationError(ValueError):
    """The requested coverage calibration has no supported finite solution."""


@dataclass(frozen=True)
class CoverageCalibration:
    num_configurations: int
    top_k: int
    target_probability: float | None
    achieved_probability: float
    poisson_rate: float
    expected_num_trials: float
    conditioning_probability: float
    workload_mode: str = "coverage_constrained"


def calibrate_poisson_coverage(
    *,
    num_configurations: int,
    top_k: int,
    target_probability: float,
    maximum_expected_num_trials: float = 1e6,
) -> CoverageCalibration:
    """Invert the existing conditioned coverage function, retaining K>=k.

    Supports k<=N, the domain of the fixed-candidate comparator. When the
    target is already reached at the mu->0 limit there is no smallest
    positive rate; report this explicitly rather than changing the count
    law into deterministic K=k or imposing an arbitrary minimum rate.
    """
    n = positive_integer(num_configurations, "num_configurations")
    k = positive_integer(top_k, "top_k")
    if k > n:
        raise ValueError("top_k must not exceed num_configurations.")
    target = open_probability(target_probability, "target_probability")
    maximum = float(maximum_expected_num_trials)
    if not math.isfinite(maximum) or maximum <= k:
        raise ValueError("maximum_expected_num_trials must be finite and exceed top_k.")
    limiting_coverage = (
        math.exp(math.lgamma(n + 1) - n * math.log(n)) if k == n else 0.0
    )
    if target <= limiting_coverage:
        raise CoverageCalibrationError(
            "Coverage target is at or below the mu->0 limiting coverage "
            f"{limiting_coverage:.8g}; there is no smallest positive calibrated rate."
        )

    def coverage(mu: float) -> float:
        return conditioned_all_configurations_coverage_probability(
            underlying_poisson_rate=mu,
            retained_count=k,
            num_configurations=n,
        )

    # Unconditioned inversion supplies a valid upper bracket.
    upper = -n * math.log(-math.expm1(math.log(target) / n))
    lower = upper / 2
    for _ in range(200):
        if coverage(lower) < target:
            break
        lower /= 2
    else:
        raise CoverageCalibrationError("Could not bracket conditioned coverage.")
    # Roundoff in the unconditioned inverse (and exp(log(upper))) can put
    # the nominal upper endpoint a few ulps below the root, especially k=1.
    # Expand the bracket before root finding, without changing the target.
    upper *= 1.0 + 1e-8
    log_mu = brentq(
        lambda x: coverage(math.exp(x)) - target,
        math.log(lower),
        math.log(upper),
        xtol=1e-12,
        rtol=1e-12,
    )
    mu = math.exp(log_mu)
    distribution = PoissonDistribution(mu)
    expected = distribution.conditional_mean(k)
    achieved = coverage(mu)
    if abs(achieved - target) > 1e-10:
        raise ArithmeticError("Conditioned coverage calibration missed its target.")
    if expected > maximum:
        raise CoverageCalibrationError(
            f"Coverage requires E[K]={expected:.8g}, above the configured limit {maximum:.8g}."
        )
    return CoverageCalibration(
        n, k, target, achieved, mu, expected, distribution.survival_probability(k)
    )


def calibrate_poisson_expected_compute(
    *, num_configurations: int, top_k: int
) -> CoverageCalibration:
    """Set E[Poisson(mu) | K>=k]=N; do not impose a coverage target."""
    n = positive_integer(num_configurations, "num_configurations")
    k = positive_integer(top_k, "top_k")
    if n <= k:
        raise CoverageCalibrationError(
            "Expected-compute matching requires N > top_k; E[K]=top_k is only "
            "a limiting degenerate count, not a positive-rate conditioned Poisson."
        )
    distribution = PoissonDistribution.from_conditional_mean(m=k, target_mean=n)
    expected = distribution.conditional_mean(k)
    if not math.isclose(expected, n, rel_tol=1e-10, abs_tol=1e-10):
        raise ArithmeticError("Conditioned Poisson calibration missed E[K]=N.")
    achieved = conditioned_all_configurations_coverage_probability(
        underlying_poisson_rate=distribution.mu,
        retained_count=k,
        num_configurations=n,
    )
    return CoverageCalibration(
        n,
        k,
        None,
        achieved,
        distribution.mu,
        expected,
        distribution.survival_probability(k),
        "expected_compute_matched",
    )


@dataclass(frozen=True)
class MethodEvaluation:
    method: str
    epsilon: float | None
    epsilon_bound: float | None
    status: str
    diagnostic: str
    expected_num_trials: float | None
    coverage_probability: float | None
    details: dict


def _order_boundary(result: ApproxDPResult) -> bool:
    # Zero epsilon cannot be improved by extending the order grid.
    return result.epsilon > 0 and (result.is_at_min_order or result.is_at_max_order)


def _evaluated(
    method: str,
    epsilon: float,
    expected_trials: float,
    coverage: float,
    details: dict,
    boundary: bool,
    reject_boundary: bool,
) -> MethodEvaluation:
    rejected = boundary and reject_boundary
    return MethodEvaluation(
        method,
        None if rejected else epsilon,
        epsilon,
        "renyi_order_boundary" if rejected else "ok",
        "Selected Renyi order is at a search boundary; extend the order grid."
        if boundary
        else "",
        expected_trials,
        coverage,
        details,
    )


def account_one_shot(
    base_curve: RdpCurve,
    *,
    num_candidates: int,
    top_k: int,
    delta_target: float,
    conversion_policy: str = "improved",
    reject_renyi_order_boundary: bool = True,
    optimizer: TCTOptimizerConfig | None = None,
) -> tuple[MethodEvaluation, MethodEvaluation, MethodEvaluation]:
    """Return TCT, full RDP, and their accepted minimum (legacy three-result API).

    The exp7 runner additionally includes account_dp_composition's bounds
    in its reported one_shot_best.
    """
    n = positive_integer(num_candidates, "num_candidates")
    k = positive_integer(top_k, "top_k")
    if k > n:
        raise ValueError("top_k must not exceed num_candidates.")
    delta_target = open_probability(delta_target, "delta_target")
    full_dp = convert_rdp_to_approx_dp_with_policy(
        scale_rdp_curve(base_curve, n),
        delta=delta_target,
        policy=conversion_policy,
    )
    full = _evaluated(
        "full_composition_one_shot",
        full_dp.epsilon,
        n,
        1.0,
        {"conversion": asdict(full_dp)},
        _order_boundary(full_dp),
        reject_renyi_order_boundary,
    )
    try:
        result = tct_topk_from_rdp(
            base_curve,
            num_candidates=n,
            top_k=k,
            delta_target=delta_target,
            conversion_policy=conversion_policy,
            optimizer=optimizer,
        )
        details = asdict(result)
        details["method_best"] = result.best.bound.method
        tct = _evaluated(
            "tct_one_shot",
            result.eps_best,
            n,
            1.0,
            details,
            _order_boundary(result.best.base_dp),
            reject_renyi_order_boundary,
        )
    except (TCTInfeasibleError, ArithmeticError) as error:
        tct = MethodEvaluation(
            "tct_one_shot",
            None,
            None,
            "infeasible"
            if isinstance(error, TCTInfeasibleError)
            else "numerical_failure",
            str(error),
            n,
            1.0,
            {},
        )
    accepted = [item for item in (tct, full) if item.epsilon is not None]
    if accepted:
        chosen = min(accepted, key=lambda item: item.epsilon)
        best = MethodEvaluation(
            "one_shot_best",
            chosen.epsilon,
            chosen.epsilon_bound,
            "ok",
            chosen.diagnostic,
            n,
            1.0,
            {
                "accountant_best": chosen.method,
                "component_statuses": {
                    item.method: item.status for item in (tct, full)
                },
            },
        )
    else:
        best = MethodEvaluation(
            "one_shot_best",
            None,
            None,
            "no_accepted_bound",
            "Neither one-shot bound passed the configured checks.",
            n,
            1.0,
            {"component_statuses": {item.method: item.status for item in (tct, full)}},
        )
    return tct, full, best


def best_accepted_bound(
    method: str, components: tuple[MethodEvaluation, ...]
) -> MethodEvaluation:
    """A derived bound, retaining the identities and statuses of its inputs."""
    if not components:
        raise ValueError("At least one component is required.")
    accepted = [item for item in components if item.epsilon is not None]
    chosen = min(accepted, key=lambda item: item.epsilon) if accepted else None
    complete = all(item.status == "ok" for item in components)
    return MethodEvaluation(
        method,
        chosen.epsilon if chosen else None,
        chosen.epsilon_bound if chosen else None,
        "ok" if chosen else "no_accepted_bound",
        (
            chosen.diagnostic
            if complete
            else "Comparison incomplete: some component bounds were rejected; minimum uses accepted components only."
        )
        if chosen
        else "No component bound passed the configured checks.",
        components[0].expected_num_trials,
        components[0].coverage_probability,
        {
            "accountant_best": chosen.method if chosen else None,
            "comparison_complete": complete,
            "component_statuses": {item.method: item.status for item in components},
        },
    )


def account_dp_composition(
    base_curve: RdpCurve,
    *,
    num_candidates: int,
    delta_target: float,
    conversion_policy: str = "improved",
    reject_renyi_order_boundary: bool = True,
    optimizer: TCTOptimizerConfig | None = None,
) -> tuple[MethodEvaluation, MethodEvaluation]:
    """Basic and advanced approximate-DP references, independent of top_k."""
    evaluations = []
    for branch in ("basic", "advanced"):
        method = f"{branch}_dp_composition_one_shot"
        try:
            result = compose_approx_dp_from_rdp(
                base_curve,
                num_candidates=num_candidates,
                delta_target=delta_target,
                method=branch,
                conversion_policy=conversion_policy,
                optimizer=optimizer,
            )
            evaluations.append(
                _evaluated(
                    method,
                    result.epsilon,
                    result.num_candidates,
                    1.0,
                    asdict(result),
                    _order_boundary(result.base_dp),
                    reject_renyi_order_boundary,
                )
            )
        except ArithmeticError as error:
            evaluations.append(
                MethodEvaluation(
                    method,
                    None,
                    None,
                    "numerical_failure",
                    str(error),
                    num_candidates,
                    1.0,
                    {},
                )
            )
    return tuple(evaluations)


def account_poisson_topk(
    base_curve: RdpCurve,
    *,
    coverage: CoverageCalibration,
    delta_target: float,
    conversion_policy: str = "improved",
    reject_renyi_order_boundary: bool = True,
) -> MethodEvaluation:
    """Use the existing top-m accountant, including conditioning for k=1."""
    result = compute_top_m_rdp_poisson(
        base_curve,
        m=coverage.top_k,
        expected_num_trials=coverage.expected_num_trials,
    )
    if not math.isclose(result.mu, coverage.poisson_rate, rel_tol=1e-9, abs_tol=1e-12):
        raise ArithmeticError(
            "Selection accountant changed the calibrated Poisson rate."
        )
    dp = convert_rdp_to_approx_dp_with_policy(
        result.rdp_curve,
        delta=delta_target,
        policy=conversion_policy,
    )
    index = dp.best_index
    source_order = float(result.envelope_source_orders[index])
    source_index = int(np.searchsorted(base_curve.orders, source_order))
    auxiliary_order = float(result.best_auxiliary_orders[source_index])
    boundary = dp.epsilon > 0 and (
        _order_boundary(dp)
        or source_order == base_curve.orders[-1]
        or auxiliary_order in (base_curve.orders[0], base_curve.orders[-1])
    )
    details = {
        "coverage": asdict(coverage),
        "conversion": asdict(dp),
        "envelope_source_order": source_order,
        "auxiliary_order": auxiliary_order,
        "hat_epsilon": float(result.hat_epsilons[source_index]),
        "hat_delta": float(result.hat_deltas[source_index]),
        "log_expected_binomial": result.log_expected_binomial,
        "base_epsilon_at_source_order": base_curve.epsilon_at(source_order),
        "selection_rdp_at_best_order": result.rdp_curve.epsilon_at(dp.best_order),
        "probability_k_zero": 0.0,
        "trial_count_law": "Poisson(mu) conditioned on K>=top_k",
    }
    return _evaluated(
        "poisson_expected_compute_matched_topk"
        if coverage.workload_mode == "expected_compute_matched"
        else "poisson_conditioned_topk",
        dp.epsilon,
        coverage.expected_num_trials,
        coverage.achieved_probability,
        details,
        boundary,
        reject_renyi_order_boundary,
    )
