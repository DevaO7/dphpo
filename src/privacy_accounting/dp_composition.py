"""Ordinary approximate-DP composition of N runs of a shared RDP mechanism.

Each run is charged once at epsilon_base, without the conditional-release
doubling or target-charging overhead of TCT. Advanced composition uses the
same Lemma A.4 upper bound as our TCT accountant.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .rdp_utils import ApproxDPResult, RdpCurve, convert_rdp_to_approx_dp_with_policy
from .tct import (
    FractionOptimization,
    TCTOptimizerConfig,
    minimize_fraction,
    open_probability,
    positive_integer,
)


@dataclass(frozen=True)
class DPCompositionResult:
    method: str
    num_candidates: int
    epsilon: float
    delta_target: float
    base_dp: ApproxDPResult
    candidate_delta_sum: float
    delta_ac: float
    base_delta_fraction: float
    optimization: FractionOptimization | None


def compose_approx_dp_from_rdp(
    base_curve: RdpCurve,
    *,
    num_candidates: int,
    delta_target: float,
    method: str,
    conversion_policy: str = "improved",
    optimizer: TCTOptimizerConfig | None = None,
) -> DPCompositionResult:
    """Basic composition or optimized advanced composition over N runs.

    Basic: N*epsilon_base, delta_base=delta_target/N.
    Advanced: N*epsilon_base**2/2 + epsilon_base*sqrt(2*N*log(1/delta_ac)),
    optimizing beta=N*delta_base/delta_target, delta_ac=(1-beta)*delta_target.
    No minimum with basic composition is taken inside the advanced result.
    """
    n = positive_integer(num_candidates, "num_candidates")
    delta = open_probability(delta_target, "delta_target")
    if method not in ("basic", "advanced"):
        raise ValueError("method must be basic or advanced.")

    def evaluate(beta: float) -> DPCompositionResult:
        base = convert_rdp_to_approx_dp_with_policy(
            base_curve, delta=delta * beta / n, policy=conversion_policy
        )
        delta_ac = 0.0 if method == "basic" else delta * (1.0 - beta)
        epsilon = (
            n * base.epsilon
            if method == "basic"
            else 0.5 * n * base.epsilon**2
            + base.epsilon * math.sqrt(2 * n * -math.log(delta_ac))
        )
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ArithmeticError("Ordinary DP composition produced a nonfinite bound.")
        return DPCompositionResult(
            method, n, epsilon, delta, base, n * base.delta, delta_ac, beta, None
        )

    if method == "basic":
        return evaluate(1.0)
    midpoint = evaluate(0.5)
    if midpoint.epsilon == 0:
        return midpoint
    optimum = minimize_fraction(
        lambda beta: evaluate(beta).epsilon, optimizer or TCTOptimizerConfig()
    )
    result = evaluate(optimum.fraction)
    return DPCompositionResult(
        method,
        n,
        result.epsilon,
        delta,
        result.base_dp,
        result.candidate_delta_sum,
        result.delta_ac,
        result.base_delta_fraction,
        optimum,
    )
