"""Optimize TCT base DP certificates obtained from one shared RDP curve.

The curve must bound each candidate's complete private output. N candidates
are evaluated once, so the sum of candidate deltas is N * delta_base.
The basic and advanced TCT branches optimize delta_base independently.
This adapter does not convert an approximate-DP guarantee back into RDP.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .rdp_utils import (
    ApproxDPResult,
    RdpCurve,
    convert_rdp_to_approx_dp_with_policy,
)
from .tct import (
    FractionOptimization,
    TCTBound,
    TCTOptimizerConfig,
    _call_parameters,
    _optimize_bound,
    minimize_fraction,
    open_probability,
    positive_integer,
)


@dataclass(frozen=True)
class TCTRDPBound:
    base_dp: ApproxDPResult
    base_delta_fraction: float
    candidate_delta_sum: float
    eps_call: float
    q: float
    log_q: float
    bound: TCTBound
    base_delta_optimization: FractionOptimization


@dataclass(frozen=True)
class TCTRDPResult:
    num_candidates: int
    top_k: int
    delta_target: float
    conversion_policy: str
    basic: TCTRDPBound
    advanced: TCTRDPBound

    @property
    def best(self) -> TCTRDPBound:
        return min((self.basic, self.advanced), key=lambda item: item.bound.epsilon)

    @property
    def eps_best(self) -> float:
        return self.best.bound.epsilon


def tct_topk_from_rdp(
    base_rdp_curve: RdpCurve,
    *,
    num_candidates: int,
    top_k: int,
    delta_target: float,
    conversion_policy: str = "improved",
    optimizer: TCTOptimizerConfig | None = None,
) -> TCTRDPResult:
    """Minimize TCT epsilon over base-delta and concentration allocations.

    Write N * delta_base = beta * delta_target, beta in (0,1).
    For each beta, convert the unchanged base curve to a DP certificate,
    then optimize the TCT concentration bound. All searches use the same
    deterministic optimizer and expose their numerical diagnostics.
    """
    if not isinstance(base_rdp_curve, RdpCurve):
        raise TypeError("base_rdp_curve must be an RdpCurve.")
    num_candidates = positive_integer(num_candidates, "num_candidates")
    top_k = positive_integer(top_k, "top_k")
    if top_k > num_candidates:
        raise ValueError("top_k must not exceed num_candidates.")
    delta_target = open_probability(delta_target, "delta_target")
    if conversion_policy not in ("standard", "improved"):
        raise ValueError("Unsupported RDP conversion policy.")
    settings = optimizer or TCTOptimizerConfig()
    branches = {}
    for method in ("basic", "advanced"):
        cache: dict[float, tuple[ApproxDPResult, TCTBound]] = {}

        def evaluate(beta: float) -> tuple[ApproxDPResult, TCTBound]:
            if beta not in cache:
                delta_base = delta_target * beta / num_candidates
                if delta_base == 0:
                    raise ArithmeticError("Base delta underflowed during optimization.")
                base_dp = convert_rdp_to_approx_dp_with_policy(
                    base_rdp_curve,
                    delta=delta_base,
                    policy=conversion_policy,
                )
                bound = _optimize_bound(
                    eps_base=base_dp.epsilon,
                    top_k=top_k,
                    delta_target=delta_target,
                    candidate_delta_sum=num_candidates * delta_base,
                    method=method,
                    optimizer=settings,
                )
                cache[beta] = base_dp, bound
            return cache[beta]

        optimum = minimize_fraction(lambda beta: evaluate(beta)[1].epsilon, settings)
        base_dp, bound = evaluate(optimum.fraction)
        eps_call, q, log_q = _call_parameters(base_dp.epsilon)
        c_delta = num_candidates * base_dp.delta
        if not math.isclose(
            math.fsum((c_delta, bound.delta_tail_bound, bound.delta_ac)),
            delta_target,
            rel_tol=1e-13,
        ):
            raise ArithmeticError("Optimized TCT delta allocation is inconsistent.")
        branches[method] = TCTRDPBound(
            base_dp,
            optimum.fraction,
            c_delta,
            eps_call,
            q,
            log_q,
            bound,
            optimum,
        )
    return TCTRDPResult(
        num_candidates,
        top_k,
        delta_target,
        conversion_policy,
        branches["basic"],
        branches["advanced"],
    )
