"""One-shot top-k accounting using Cohen & Lyu, NeurIPS 2023.

The Target-Charging Technique for Privacy Analysis across Interactive
Computations, https://proceedings.neurips.cc/paper_files/paper/2023/file/
c3fe2a07ec47b89c50e89706d2e23358-Paper-Conference.pdf

Lemma 2.6 gives k hits with eps_call = 2 eps_base. Lemma 2.3 gives
q = 1 / (1 + exp(eps_call)). Theorem B.4 bounds the effective access
count and failure probability. Advanced composition uses Lemma A.4,
including the factor 2 inside the square root, as instructed by B.4's
proof; the displayed B.4 expression omits that factor.

This module takes DP certificates, not training implementations or RDP
curves. No conditional-release simulation or candidate sampling is run.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit


class TCTInfeasibleError(ValueError):
    """The supplied certificates leave no positive TCT failure budget."""


class TCTOptimizationError(ArithmeticError):
    """The numerical optimization could not establish an interior solution."""


def positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return int(value)


def open_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1.")
    return value


@dataclass(frozen=True)
class TCTOptimizerConfig:
    relative_tolerance: float = 1e-8
    max_iterations: int = 200
    grid_points: int = 65
    logit_limit: float = 30.0

    def __post_init__(self) -> None:
        positive_integer(self.max_iterations, "max_iterations")
        positive_integer(self.grid_points, "grid_points")
        if self.grid_points < 9:
            raise ValueError("grid_points must be at least 9.")
        open_probability(self.relative_tolerance, "relative_tolerance")
        if not math.isfinite(self.logit_limit) or not 4 <= self.logit_limit <= 32:
            raise ValueError("logit_limit must be between 4 and 32.")


@dataclass(frozen=True)
class FractionOptimization:
    fraction: float
    objective: float
    logit: float
    evaluations: int
    refined_intervals: int
    search_lower_logit: float
    search_upper_logit: float
    at_search_boundary: bool


def minimize_fraction(
    objective: Callable[[float], float],
    config: TCTOptimizerConfig,
) -> FractionOptimization:
    """Search a fraction deterministically; refine all grid-local minima.

    No global-unimodality assumption is made. A boundary solution raises an
    explicit error instead of silently treating a bounded search as optimal.
    Every evaluated objective must itself be a valid privacy upper bound."""
    evaluations = 0

    def evaluate(x: float) -> float:
        nonlocal evaluations
        evaluations += 1
        value = float(objective(float(expit(x))))
        if math.isnan(value) or value < 0:
            raise ArithmeticError("Invalid optimization objective.")
        return value

    grid = np.linspace(-config.logit_limit, config.logit_limit, config.grid_points)
    values = [evaluate(float(x)) for x in grid]
    if not any(math.isfinite(v) for v in values):
        raise TCTOptimizationError("No finite TCT bound on the search grid.")
    candidates = [(v, float(x)) for v, x in zip(values, grid)]
    intervals = 0
    # Include the best grid basin even for a flat objective.
    best_index = min(range(len(values)), key=lambda i: (values[i], abs(grid[i])))
    indices = {
        i
        for i in range(1, len(grid) - 1)
        if values[i] <= min(values[i - 1], values[i + 1])
        and values[i] < max(values[i - 1], values[i + 1])
    }
    if 0 < best_index < len(grid) - 1:
        indices.add(best_index)
    for i in sorted(indices):
        result = minimize_scalar(
            evaluate,
            bounds=(float(grid[i - 1]), float(grid[i + 1])),
            method="bounded",
            options={
                "xatol": config.relative_tolerance,
                "maxiter": config.max_iterations,
            },
        )
        if not result.success:
            raise TCTOptimizationError(f"Fraction refinement failed: {result.message}")
        intervals += 1
        candidates.append((float(result.fun), float(result.x)))
    value, x = min(candidates, key=lambda pair: (pair[0], abs(pair[1])))
    at_boundary = abs(x) >= config.logit_limit - config.relative_tolerance
    if at_boundary:
        raise TCTOptimizationError(
            "Best allocation is at the logit search boundary; widen logit_limit "
            "or inspect the privacy curve."
        )
    return FractionOptimization(
        fraction=float(expit(x)),
        objective=value,
        logit=x,
        evaluations=evaluations,
        refined_intervals=intervals,
        search_lower_logit=-config.logit_limit,
        search_upper_logit=config.logit_limit,
        at_search_boundary=at_boundary,
    )


@dataclass(frozen=True)
class TCTBound:
    method: str
    epsilon: float
    alpha: float
    effective_accesses: float
    delta_tail_bound: float
    delta_ac: float
    delta_used: float
    tail_fraction: float
    optimization: FractionOptimization | None = None


@dataclass(frozen=True)
class TCTPrivacyResult:
    num_candidates: int
    top_k: int
    eps_base: float
    eps_call: float
    q: float
    log_q: float
    candidate_delta_sum: float
    delta_target: float
    basic: TCTBound
    advanced: TCTBound

    @property
    def best(self) -> TCTBound:
        return min((self.basic, self.advanced), key=lambda result: result.epsilon)

    @property
    def eps_basic(self) -> float:
        return self.basic.epsilon

    @property
    def eps_advanced(self) -> float:
        return self.advanced.epsilon

    @property
    def eps_best(self) -> float:
        return self.best.epsilon

    @property
    def method_best(self) -> str:
        return self.best.method


def _call_parameters(eps_base: float) -> tuple[float, float, float]:
    eps_base = float(eps_base)
    if not math.isfinite(eps_base) or eps_base < 0:
        raise ValueError("eps_base must be finite and nonnegative.")
    eps_call = 2 * eps_base
    if not math.isfinite(eps_call):
        raise ArithmeticError("eps_call is outside the floating-point range.")
    log_q = -float(np.logaddexp(0.0, eps_call))
    return eps_call, math.exp(log_q), log_q


def _bound_at_tail_fraction(
    *,
    eps_base: float,
    top_k: int,
    delta_target: float,
    candidate_delta_sum: float,
    method: str,
    tail_fraction: float,
) -> TCTBound:
    eps_call, _, log_q = _call_parameters(eps_base)
    remaining = delta_target - candidate_delta_sum
    if remaining <= 8 * math.ulp(delta_target):
        raise TCTInfeasibleError("delta_target must exceed sum(candidate_deltas).")
    tail = remaining * tail_fraction
    if tail <= 0:
        raise ArithmeticError("Tail budget underflowed.")
    # Invert exp(-k alpha^2 / (2(1+alpha))) = tail in closed form.
    u = -math.log(tail) / top_k
    alpha = u + math.sqrt(u) * math.sqrt(u + 2)
    log_r = math.log1p(alpha) + math.log(top_k) - log_q
    if log_r > math.log(np.finfo(float).max):
        raise ArithmeticError("TCT effective access count overflowed.")
    r = math.exp(log_r)
    delta_ac = remaining * (1 - tail_fraction) if method == "advanced" else 0.0
    if method == "basic":
        epsilon = r * eps_call
    elif method == "advanced":
        if delta_ac <= 0:
            raise ValueError("Advanced composition requires positive delta_ac.")
        epsilon = 0.5 * r * eps_call**2 + eps_call * math.sqrt(
            2 * r * -math.log(delta_ac)
        )
    else:
        raise ValueError("method must be 'basic' or 'advanced'.")
    if not math.isfinite(epsilon):
        raise ArithmeticError("TCT epsilon overflowed.")
    used = math.fsum((candidate_delta_sum, tail, delta_ac))
    if used > delta_target and not math.isclose(used, delta_target, rel_tol=1e-14):
        raise ArithmeticError("TCT failure allocation exceeds delta_target.")
    return TCTBound(method, epsilon, alpha, r, tail, delta_ac, used, tail_fraction)


def _optimize_bound(
    *,
    eps_base: float,
    top_k: int,
    delta_target: float,
    candidate_delta_sum: float,
    method: str,
    optimizer: TCTOptimizerConfig,
) -> TCTBound:
    from dataclasses import replace

    def at_fraction(fraction: float) -> TCTBound:
        return _bound_at_tail_fraction(
            eps_base=eps_base,
            top_k=top_k,
            delta_target=delta_target,
            candidate_delta_sum=candidate_delta_sum,
            method=method,
            tail_fraction=fraction,
        )

    if method == "basic":
        return at_fraction(1.0)
    if eps_base == 0:
        return at_fraction(0.5)
    optimum = minimize_fraction(lambda f: at_fraction(f).epsilon, optimizer)
    return replace(at_fraction(optimum.fraction), optimization=optimum)


def tct_topk_privacy(
    *,
    eps_base: float,
    top_k: int,
    delta_target: float,
    num_candidates: int | None = None,
    delta_base: float | None = None,
    candidate_deltas: Sequence[float] | None = None,
    optimizer: TCTOptimizerConfig | None = None,
) -> TCTPrivacyResult:
    """Account for N independent private candidates, each evaluated once.

    The common epsilon bound must hold for the complete (model, score) output.
    Exactly one of delta_base and candidate_deltas is required. A delta is
    charged once per original candidate, including candidates not released;
    revision calls do not add further candidate deltas. Return both optimized
    bounds, with the conservative Lemma A.4 advanced-composition constant.

    Infeasible means that these supplied certificates leave no positive tail
    budget, not that the underlying mechanism cannot satisfy the target."""
    top_k = positive_integer(top_k, "top_k")
    delta_target = open_probability(delta_target, "delta_target")
    eps_call, q, log_q = _call_parameters(eps_base)
    if (delta_base is None) == (candidate_deltas is None):
        raise ValueError("Supply exactly one of delta_base or candidate_deltas.")
    if candidate_deltas is not None:
        deltas = tuple(float(d) for d in candidate_deltas)
        if num_candidates is None:
            num_candidates = len(deltas)
        elif positive_integer(num_candidates, "num_candidates") != len(deltas):
            raise ValueError("num_candidates does not match candidate_deltas.")
    else:
        if num_candidates is None:
            raise ValueError("num_candidates is required with delta_base.")
        deltas = (float(delta_base),)
    num_candidates = positive_integer(num_candidates, "num_candidates")
    if top_k > num_candidates:
        raise ValueError("top_k must not exceed num_candidates.")
    if any(not math.isfinite(d) or not 0 <= d < 1 for d in deltas):
        raise ValueError("Candidate deltas must be finite and in [0, 1).")
    c_delta = (
        math.fsum(deltas)
        if candidate_deltas is not None
        else num_candidates * deltas[0]
    )
    if delta_target - c_delta <= 8 * math.ulp(delta_target):
        raise TCTInfeasibleError("delta_target must exceed sum(candidate_deltas).")
    settings = optimizer or TCTOptimizerConfig()
    bounds = {
        method: _optimize_bound(
            eps_base=float(eps_base),
            top_k=top_k,
            delta_target=delta_target,
            candidate_delta_sum=c_delta,
            method=method,
            optimizer=settings,
        )
        for method in ("basic", "advanced")
    }
    return TCTPrivacyResult(
        num_candidates,
        top_k,
        float(eps_base),
        eps_call,
        q,
        log_q,
        c_delta,
        delta_target,
        bounds["basic"],
        bounds["advanced"],
    )
