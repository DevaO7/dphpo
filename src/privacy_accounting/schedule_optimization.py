"""Pure schedule construction for fixed-budget geometric DP-SHA.

This module deliberately contains no Hydra, persistence, plotting, or privacy
calibration code.  It defines the data-independent schedule family used by
the fixed expected-budget planner and evaluates its non-privacy constraints.

For a schedule with ``L >= 2`` stages, initial expected workload ``w_1``,
retention factor ``q``, minimum resource ``r_min``, and required final
resource ``R``, the intended workloads and cumulative resources are

    w_l = w_1 q**(l - 1),
    r_l = r_min (R / r_min)**((l - 1) / (L - 1)).

Every non-final stage uses a Poisson count conditioned on ``K >= m_l`` with
``m_l = round(q w_l)``.  The final stage uses an unconditioned Poisson top-1
count.  The whole schedule is fixed before any count is sampled; realized
compute is never recycled into later stages. The ``L=1`` boundary is the
Papernot--Steinke Poisson mechanism: one unconditioned top-1 stage at
``r_1=R`` with no ``q`` or resource-growth factor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Iterable

from scipy.optimize import brentq
from scipy.special import gammainc

from .dpsgd import compute_dpsgd_rdp
from .poisson import PoissonDistribution
from .rdp_utils import (
    ApproxDPResult,
    RdpCurve,
    convert_rdp_to_approx_dp,
    normalize_requested_integer_orders,
    scale_rdp_curve,
)
from .selection_accounting import (
    PoissonNStageResult,
    compute_n_stage_rdp_poisson,
)
from .validation import (
    validate_positive_float,
    validate_positive_integer,
)


_PROBABILITY_TOLERANCE = 1e-12
_COVERAGE_CRITERIA = {
    "all_configurations",
    "at_least_one_good",
}


def _validate_open_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be finite and satisfy 0 < {name} < 1.")
    return value


def _validate_retention_factor(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(
            "retention_factor must be finite and satisfy 0 < q < 1."
        )
    return value


def _normalize_coverage_criterion(value: str) -> str:
    criterion = str(value).strip().lower()
    if criterion not in _COVERAGE_CRITERIA:
        available = ", ".join(sorted(_COVERAGE_CRITERIA))
        raise ValueError(
            f"coverage_criterion must be one of: {available}."
        )
    return criterion


def _validate_rounding_mode(mode: str, name: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in {"floor", "ceil", "nearest"}:
        raise ValueError(
            f"{name} must be 'floor', 'ceil', or 'nearest'."
        )
    return mode


def _round_positive(value: float, *, mode: str, name: str) -> int:
    value = validate_positive_float(value, name)
    mode = _validate_rounding_mode(mode, f"{name} rounding mode")
    if mode == "floor":
        rounded = math.floor(value)
    elif mode == "ceil":
        rounded = math.ceil(value)
    else:
        rounded = math.floor(value + 0.5)
    return max(int(rounded), 1)


def _validate_configuration_counts(
    *,
    num_configurations: int,
    num_good_configurations: int | None,
    coverage_criterion: str = "at_least_one_good",
) -> tuple[int, int | None]:
    coverage_criterion = _normalize_coverage_criterion(coverage_criterion)
    num_configurations = validate_positive_integer(
        num_configurations,
        "num_configurations",
    )
    if coverage_criterion == "all_configurations":
        if num_good_configurations is not None:
            raise ValueError(
                "num_good_configurations must be None when "
                "coverage_criterion='all_configurations'."
            )
        return num_configurations, None
    if num_good_configurations is None:
        raise ValueError(
            "num_good_configurations is required when coverage_criterion="
            "'at_least_one_good'."
        )
    num_good_configurations = validate_positive_integer(
        num_good_configurations,
        "num_good_configurations",
    )
    if num_good_configurations > num_configurations:
        raise ValueError(
            "num_good_configurations cannot exceed num_configurations."
        )
    return num_configurations, num_good_configurations


@dataclass(frozen=True, slots=True)
class GeometricScheduleStage:
    """One predeclared stage in a geometric expected-workload schedule."""

    stage: int
    expected_num_trials: float
    retained_count: int
    cumulative_resource: int
    incremental_resource: int
    count_semantics: str
    underlying_poisson_rate: float
    conditioning_probability: float | None
    probability_k_zero: float
    expected_compute: float


@dataclass(frozen=True, slots=True)
class GeometricSchedule:
    """A complete fixed schedule whose final resource is exactly ``R``."""

    num_stages: int
    initial_expected_num_trials: float
    retention_factor: float | None
    resource_growth_factor: float | None
    minimum_resource: int
    required_final_resource: int
    survivor_rounding: str
    resource_rounding: str
    stages: tuple[GeometricScheduleStage, ...]
    expected_compute: float


@dataclass(frozen=True, slots=True)
class InitialBreadthSolution:
    """Smallest Stage-1 expected workload meeting the two probability goals."""

    coverage_criterion: str
    initial_expected_num_trials: float
    stage_1_retained_count: int
    stage_1_underlying_poisson_rate: float
    initial_coverage_probability: float
    final_expected_num_trials: float
    final_nonempty_probability: float

    @property
    def initial_good_coverage_probability(self) -> float:
        """Backward-compatible name for the generic coverage probability."""
        return self.initial_coverage_probability


@dataclass(frozen=True, slots=True)
class ScheduleConstraintResult:
    """Evaluation of all non-privacy constraints for one valid schedule."""

    coverage_criterion: str
    is_feasible: bool
    rejection_reasons: tuple[str, ...]
    expected_compute: float
    expected_compute_budget: float
    expected_compute_margin: float
    initial_coverage_probability: float
    target_initial_coverage: float
    initial_coverage_margin: float
    final_nonempty_probability: float
    target_final_nonempty_probability: float
    final_nonempty_probability_margin: float

    @property
    def initial_good_coverage_probability(self) -> float:
        """Backward-compatible alias used by earlier pilot code."""
        return self.initial_coverage_probability

    @property
    def target_initial_good_coverage(self) -> float:
        """Backward-compatible alias used by earlier pilot code."""
        return self.target_initial_coverage

    @property
    def initial_good_coverage_margin(self) -> float:
        """Backward-compatible alias used by earlier pilot code."""
        return self.initial_coverage_margin


@dataclass(frozen=True, slots=True)
class GeometricScheduleCandidate:
    """One enumerated ``(L, q)`` candidate and its feasibility diagnostics."""

    num_stages: int
    retention_factor: float | None
    schedule: GeometricSchedule | None
    constraints: ScheduleConstraintResult | None
    rejection_reasons: tuple[str, ...]
    diagnostic: str | None = None

    @property
    def is_feasible(self) -> bool:
        return (
            self.schedule is not None
            and self.constraints is not None
            and self.constraints.is_feasible
        )


@dataclass(frozen=True, slots=True)
class GeometricScheduleSearchResult:
    """Deterministically ordered results from a structural schedule search."""

    candidates: tuple[GeometricScheduleCandidate, ...]

    @property
    def feasible_candidates(self) -> tuple[GeometricScheduleCandidate, ...]:
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.is_feasible
        )


@dataclass(frozen=True, slots=True)
class SchedulePrivacyEvaluation:
    """N-stage privacy result for one schedule and common noise multiplier."""

    noise_multiplier: float
    stage_base_rdp_curves: tuple[RdpCurve, ...]
    n_stage_result: PoissonNStageResult
    approximate_dp: ApproxDPResult


@dataclass(frozen=True, slots=True)
class ScheduleSigmaCalibrationResult:
    """Smallest configured common sigma satisfying a target epsilon."""

    target_epsilon: float
    achieved_epsilon: float
    delta: float
    noise_multiplier: float
    best_renyi_order: float
    relative_sigma_tolerance: float
    bisection_iterations: int
    accountant_evaluations: int
    at_minimum_sigma: bool
    infeasible_lower_sigma: float | None
    infeasible_lower_epsilon: float | None
    feasible_upper_sigma: float
    feasible_upper_epsilon: float
    privacy_evaluation: SchedulePrivacyEvaluation


class SchedulePrivacyCalibrationError(ValueError):
    """A classified privacy-calibration failure for one schedule."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = str(reason)


@dataclass(frozen=True, slots=True)
class PrivacyCalibratedScheduleCandidate:
    """A structural candidate augmented with privacy-calibration results."""

    structural_candidate: GeometricScheduleCandidate
    calibration: ScheduleSigmaCalibrationResult | None
    rejection_reasons: tuple[str, ...]
    diagnostic: str | None = None

    @property
    def is_feasible(self) -> bool:
        return (
            self.structural_candidate.is_feasible
            and self.calibration is not None
            and not self.rejection_reasons
        )


@dataclass(frozen=True, slots=True)
class PrivacyCalibratedScheduleSearchResult:
    """Privacy-calibrated candidates in structural-search order."""

    candidates: tuple[PrivacyCalibratedScheduleCandidate, ...]

    @property
    def feasible_candidates(
        self,
    ) -> tuple[PrivacyCalibratedScheduleCandidate, ...]:
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.is_feasible
        )


def unconditioned_coverage_probability(
    *,
    underlying_poisson_rate: float,
    num_configurations: int,
    coverage_criterion: str,
    num_good_configurations: int | None = None,
) -> float:
    r"""Evaluate one of the two declared unconditioned coverage events.

    For ``all_configurations``, this is the expression derived in
    ``main.tex``:

    .. math::

        \alpha_P = (1 - \exp(-\theta/N))^N.

    For ``at_least_one_good``, Poisson splitting gives

    .. math::

        1 - \exp(-\theta G/N).
    """
    underlying_poisson_rate = validate_positive_float(
        underlying_poisson_rate,
        "underlying_poisson_rate",
    )
    coverage_criterion = _normalize_coverage_criterion(
        coverage_criterion
    )
    num_configurations, num_good_configurations = (
        _validate_configuration_counts(
            num_configurations=num_configurations,
            num_good_configurations=num_good_configurations,
            coverage_criterion=coverage_criterion,
        )
    )
    if coverage_criterion == "all_configurations":
        probability = (-math.expm1(
            -underlying_poisson_rate / num_configurations
        )) ** num_configurations
    else:
        if num_good_configurations is None:
            raise RuntimeError("Validated good-configuration count is absent.")
        probability = -math.expm1(
            -underlying_poisson_rate
            * num_good_configurations
            / num_configurations
        )
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ArithmeticError(
            "Unconditioned coverage probability fell outside [0, 1]."
        )
    return float(probability)


def conditioned_all_configurations_coverage_probability(
    *,
    underlying_poisson_rate: float,
    retained_count: int,
    num_configurations: int,
) -> float:
    r"""Return ``alpha_m`` from the coverage subsection of ``main.tex``.

    The trial count is ``Poisson(theta)`` conditioned on ``K >= m`` and
    configurations are sampled uniformly with replacement.  The numerator
    is the probability of both covering every configuration and satisfying
    the conditioning event.  Inclusion--exclusion over missing
    configurations rewrites the infinite expression in ``main.tex`` as a
    finite sum of Poisson survival probabilities.
    """
    underlying_poisson_rate = validate_positive_float(
        underlying_poisson_rate,
        "underlying_poisson_rate",
    )
    retained_count = validate_positive_integer(
        retained_count,
        "retained_count",
    )
    num_configurations, _ = _validate_configuration_counts(
        num_configurations=num_configurations,
        num_good_configurations=None,
        coverage_criterion="all_configurations",
    )
    distribution = PoissonDistribution(underlying_poisson_rate)
    conditioning_probability = distribution.survival_probability(
        retained_count
    )
    unconditional_coverage = unconditioned_coverage_probability(
        underlying_poisson_rate=underlying_poisson_rate,
        num_configurations=num_configurations,
        coverage_criterion="all_configurations",
    )
    if retained_count <= num_configurations:
        # Covering all N configurations implies K >= N >= m.
        probability = unconditional_coverage / conditioning_probability
    else:
        # For a fixed set of j missing configurations, Poisson splitting
        # gives a zero-count factor exp(-theta*j/N); all remaining draws
        # form Poisson(theta*(N-j)/N). Requiring at least m total draws and
        # applying inclusion-exclusion yields the numerator below.
        terms = []
        for missing_count in range(num_configurations + 1):
            remaining_fraction = (
                num_configurations - missing_count
            ) / num_configurations
            remaining_rate = underlying_poisson_rate * remaining_fraction
            remaining_survival = (
                0.0
                if remaining_rate == 0.0
                else float(gammainc(retained_count, remaining_rate))
            )
            term = (
                math.comb(num_configurations, missing_count)
                * math.exp(
                    -underlying_poisson_rate
                    * missing_count
                    / num_configurations
                )
                * remaining_survival
            )
            terms.append(-term if missing_count % 2 else term)
        numerator = math.fsum(terms)
        probability = numerator / conditioning_probability

    if (
        not math.isfinite(probability)
        or probability < -_PROBABILITY_TOLERANCE
        or probability > 1.0 + _PROBABILITY_TOLERANCE
    ):
        raise ArithmeticError(
            "Conditioned all-configurations coverage fell outside [0, 1]."
        )
    return min(max(float(probability), 0.0), 1.0)


def conditioned_good_configuration_coverage_probability(
    *,
    underlying_poisson_rate: float,
    retained_count: int,
    num_configurations: int,
    num_good_configurations: int,
) -> float:
    r"""Return the chance of sampling at least one of ``G`` good HPs.

    The trial count is

    .. math::

        K \sim \operatorname{Poisson}(\theta) \mid K \ge m,

    and every trial samples uniformly with replacement from ``N``
    configurations, of which ``G`` are good.  Poisson splitting gives

    .. math::

        1 - e^{-\theta G/N}
        \frac{\Pr[\operatorname{Poisson}(\theta(1-G/N)) \ge m]}
             {\Pr[\operatorname{Poisson}(\theta) \ge m]}.

    This is not the probability that all ``N`` configurations are covered.
    """
    underlying_poisson_rate = validate_positive_float(
        underlying_poisson_rate,
        "underlying_poisson_rate",
    )
    retained_count = validate_positive_integer(
        retained_count,
        "retained_count",
    )
    num_configurations, num_good_configurations = (
        _validate_configuration_counts(
            num_configurations=num_configurations,
            num_good_configurations=num_good_configurations,
            coverage_criterion="at_least_one_good",
        )
    )

    if num_good_configurations == num_configurations:
        return 1.0

    good_fraction = num_good_configurations / num_configurations
    total_distribution = PoissonDistribution(underlying_poisson_rate)
    bad_rate = underlying_poisson_rate * (1.0 - good_fraction)
    # Unlike PoissonDistribution.survival_probability(), zero is a valid
    # numerical result here: it means that the no-good event is negligible
    # at floating-point precision and hence coverage is one.  This matters
    # when almost every configuration is good and m is large.
    bad_survival_probability = float(gammainc(retained_count, bad_rate))
    probability_no_good = (
        math.exp(-underlying_poisson_rate * good_fraction)
        * bad_survival_probability
        / total_distribution.survival_probability(retained_count)
    )
    probability = 1.0 - probability_no_good
    if (
        not math.isfinite(probability)
        or probability < -_PROBABILITY_TOLERANCE
        or probability > 1.0 + _PROBABILITY_TOLERANCE
    ):
        raise ArithmeticError(
            "Conditioned good-configuration coverage fell outside [0, 1]."
        )
    return min(max(probability, 0.0), 1.0)


def conditioned_coverage_probability(
    *,
    underlying_poisson_rate: float,
    retained_count: int,
    num_configurations: int,
    coverage_criterion: str,
    num_good_configurations: int | None = None,
) -> float:
    """Dispatch to the declared conditioned Stage-1 coverage event."""
    coverage_criterion = _normalize_coverage_criterion(coverage_criterion)
    if coverage_criterion == "all_configurations":
        if num_good_configurations is not None:
            raise ValueError(
                "num_good_configurations must be None for "
                "all_configurations coverage."
            )
        return conditioned_all_configurations_coverage_probability(
            underlying_poisson_rate=underlying_poisson_rate,
            retained_count=retained_count,
            num_configurations=num_configurations,
        )
    if num_good_configurations is None:
        raise ValueError(
            "num_good_configurations is required for at_least_one_good "
            "coverage."
        )
    return conditioned_good_configuration_coverage_probability(
        underlying_poisson_rate=underlying_poisson_rate,
        retained_count=retained_count,
        num_configurations=num_configurations,
        num_good_configurations=num_good_configurations,
    )


def build_geometric_schedule(
    *,
    initial_expected_num_trials: float,
    retention_factor: float | None,
    num_stages: int,
    minimum_resource: int,
    required_final_resource: int,
    survivor_rounding: str = "floor",
    resource_rounding: str = "nearest",
) -> GeometricSchedule:
    r"""Build a fixed schedule with a Papernot ``L=1`` boundary case.

    For ``L >= 2``, this is the geometric DP-SHA family with
    ``r_1=r_min`` and ``r_L=R``.  For ``L=1``, the only stage is the final
    unconditioned-Poisson top-1 mechanism, so ``r_1=R`` and both ``q`` and
    ``rho`` are not applicable.  This is Papernot--Steinke Poisson tuning.
    """
    initial_expected_num_trials = validate_positive_float(
        initial_expected_num_trials,
        "initial_expected_num_trials",
    )
    num_stages = validate_positive_integer(num_stages, "num_stages")
    if num_stages == 1:
        if retention_factor is not None:
            raise ValueError(
                "retention_factor must be None for the L=1 Papernot "
                "boundary case."
            )
    else:
        if retention_factor is None:
            raise ValueError(
                "retention_factor is required for schedules with L >= 2."
            )
        retention_factor = _validate_retention_factor(retention_factor)
    if num_stages < 1:
        raise ValueError(
            "num_stages must be positive."
        )
    minimum_resource = validate_positive_integer(
        minimum_resource,
        "minimum_resource",
    )
    required_final_resource = validate_positive_integer(
        required_final_resource,
        "required_final_resource",
    )
    if num_stages >= 2 and required_final_resource <= minimum_resource:
        raise ValueError(
            "required_final_resource must be greater than minimum_resource."
        )
    if (
        num_stages >= 2
        and required_final_resource - minimum_resource < num_stages - 1
    ):
        raise ValueError(
            "The integer resource interval is too small to assign strictly "
            "increasing resources to every stage."
        )
    survivor_rounding = _validate_rounding_mode(
        survivor_rounding,
        "survivor_rounding",
    )
    resource_rounding = _validate_rounding_mode(
        resource_rounding,
        "resource_rounding",
    )

    resource_growth_factor = (
        None
        if num_stages == 1
        else (required_final_resource / minimum_resource)
        ** (1.0 / (num_stages - 1))
    )
    stages: list[GeometricScheduleStage] = []
    previous_resource = 0
    previous_retained_count: int | None = None

    for stage_index in range(num_stages):
        stage_number = stage_index + 1
        expected_num_trials = (
            initial_expected_num_trials
            if retention_factor is None
            else initial_expected_num_trials
            * retention_factor**stage_index
        )

        if num_stages == 1:
            cumulative_resource = required_final_resource
        elif stage_number == 1:
            cumulative_resource = minimum_resource
        elif stage_number == num_stages:
            cumulative_resource = required_final_resource
        else:
            cumulative_resource = _round_positive(
                minimum_resource
                * resource_growth_factor**stage_index,
                mode=resource_rounding,
                name="cumulative_resource",
            )
        if cumulative_resource <= previous_resource:
            raise ValueError(
                "Rounded cumulative resources must strictly increase at "
                f"Stage {stage_number}; received {cumulative_resource} "
                f"after {previous_resource}."
            )

        is_final = stage_number == num_stages
        if is_final:
            retained_count = 1
            count_semantics = "unconditioned_poisson_mean"
            distribution = PoissonDistribution.from_mean(
                target_mean=expected_num_trials
            )
            conditioning_probability = None
            probability_k_zero = distribution.probability_zero()
        else:
            retained_count = _round_positive(
                retention_factor * expected_num_trials,
                mode=survivor_rounding,
                name="retained_count",
            )
            if expected_num_trials <= retained_count:
                raise ValueError(
                    f"Stage {stage_number} requires conditional expected "
                    "workload bar_mu_l > m_l, but received "
                    f"bar_mu_l={expected_num_trials:g} and "
                    f"m_l={retained_count}."
                )
            count_semantics = "conditional_mean_given_k_ge_m"
            distribution = PoissonDistribution.from_conditional_mean(
                m=retained_count,
                target_mean=expected_num_trials,
            )
            conditioning_probability = distribution.survival_probability(
                retained_count
            )
            probability_k_zero = 0.0

        if (
            previous_retained_count is not None
            and retained_count > previous_retained_count
        ):
            raise ValueError(
                "Retained counts must be nonincreasing, but Stage "
                f"{stage_number} retains {retained_count} after "
                f"{previous_retained_count}."
            )

        incremental_resource = cumulative_resource - previous_resource
        expected_compute = expected_num_trials * incremental_resource
        stages.append(
            GeometricScheduleStage(
                stage=stage_number,
                expected_num_trials=float(expected_num_trials),
                retained_count=retained_count,
                cumulative_resource=cumulative_resource,
                incremental_resource=incremental_resource,
                count_semantics=count_semantics,
                underlying_poisson_rate=distribution.mu,
                conditioning_probability=conditioning_probability,
                probability_k_zero=probability_k_zero,
                expected_compute=float(expected_compute),
            )
        )
        previous_resource = cumulative_resource
        previous_retained_count = retained_count

    expected_compute = math.fsum(stage.expected_compute for stage in stages)
    return GeometricSchedule(
        num_stages=num_stages,
        initial_expected_num_trials=float(initial_expected_num_trials),
        retention_factor=retention_factor,
        resource_growth_factor=(
            None
            if resource_growth_factor is None
            else float(resource_growth_factor)
        ),
        minimum_resource=(
            required_final_resource if num_stages == 1 else minimum_resource
        ),
        required_final_resource=required_final_resource,
        survivor_rounding=survivor_rounding,
        resource_rounding=resource_rounding,
        stages=tuple(stages),
        expected_compute=float(expected_compute),
    )


def find_minimum_feasible_papernot_breadth(
    *,
    num_configurations: int,
    num_good_configurations: int | None,
    coverage_criterion: str,
    target_initial_coverage: float,
    target_final_nonempty_probability: float,
    maximum_initial_expected_num_trials: float = 1_000_000.0,
) -> InitialBreadthSolution:
    """Find the minimum unconditioned Poisson rate for the ``L=1`` case."""
    coverage_criterion = _normalize_coverage_criterion(coverage_criterion)
    num_configurations, num_good_configurations = (
        _validate_configuration_counts(
            num_configurations=num_configurations,
            num_good_configurations=num_good_configurations,
            coverage_criterion=coverage_criterion,
        )
    )
    target_initial_coverage = _validate_open_probability(
        target_initial_coverage,
        "target_initial_coverage",
    )
    target_final_nonempty_probability = _validate_open_probability(
        target_final_nonempty_probability,
        "target_final_nonempty_probability",
    )
    maximum_initial_expected_num_trials = validate_positive_float(
        maximum_initial_expected_num_trials,
        "maximum_initial_expected_num_trials",
    )
    if coverage_criterion == "all_configurations":
        probability_per_configuration = math.exp(
            math.log(target_initial_coverage) / num_configurations
        )
        coverage_rate = -num_configurations * math.log1p(
            -probability_per_configuration
        )
    else:
        if num_good_configurations is None:
            raise RuntimeError("Validated good-configuration count is absent.")
        coverage_rate = (
            -num_configurations
            / num_good_configurations
            * math.log1p(-target_initial_coverage)
        )
    final_activity_rate = -math.log1p(
        -target_final_nonempty_probability
    )
    poisson_rate = max(coverage_rate, final_activity_rate)
    if poisson_rate > maximum_initial_expected_num_trials:
        raise ValueError(
            "No Papernot expected breadth can satisfy the coverage and "
            "final-activity constraints within the configured maximum."
        )
    coverage_probability = unconditioned_coverage_probability(
        underlying_poisson_rate=poisson_rate,
        num_configurations=num_configurations,
        coverage_criterion=coverage_criterion,
        num_good_configurations=num_good_configurations,
    )
    final_nonempty_probability = -math.expm1(-poisson_rate)
    return InitialBreadthSolution(
        coverage_criterion=coverage_criterion,
        initial_expected_num_trials=float(poisson_rate),
        stage_1_retained_count=1,
        stage_1_underlying_poisson_rate=float(poisson_rate),
        initial_coverage_probability=float(coverage_probability),
        final_expected_num_trials=float(poisson_rate),
        final_nonempty_probability=float(final_nonempty_probability),
    )


def find_minimum_feasible_initial_breadth(
    *,
    retention_factor: float,
    num_stages: int,
    num_configurations: int,
    num_good_configurations: int | None,
    target_initial_good_coverage: float,
    target_final_nonempty_probability: float,
    coverage_criterion: str = "at_least_one_good",
    maximum_initial_expected_num_trials: float = 1_000_000.0,
    relative_tolerance: float = 1e-10,
) -> InitialBreadthSolution:
    """Find the smallest ``w_1`` meeting coverage and final-activity goals.

    The exact interval search currently supports the planner's canonical
    survivor rule ``m_l = max(floor(q w_l), 1)``.  Intervals on which
    ``m_1`` is fixed are searched in increasing order, avoiding a false
    continuity assumption across integer survivor-count changes.
    """
    retention_factor = _validate_retention_factor(retention_factor)
    num_stages = validate_positive_integer(num_stages, "num_stages")
    if num_stages < 2:
        raise ValueError("num_stages must be at least 2.")
    coverage_criterion = _normalize_coverage_criterion(coverage_criterion)
    num_configurations, num_good_configurations = (
        _validate_configuration_counts(
            num_configurations=num_configurations,
            num_good_configurations=num_good_configurations,
            coverage_criterion=coverage_criterion,
        )
    )
    target_initial_good_coverage = _validate_open_probability(
        target_initial_good_coverage,
        "target_initial_good_coverage",
    )
    target_final_nonempty_probability = _validate_open_probability(
        target_final_nonempty_probability,
        "target_final_nonempty_probability",
    )
    maximum_initial_expected_num_trials = validate_positive_float(
        maximum_initial_expected_num_trials,
        "maximum_initial_expected_num_trials",
    )
    relative_tolerance = validate_positive_float(
        relative_tolerance,
        "relative_tolerance",
    )
    if relative_tolerance >= 1.0:
        raise ValueError("relative_tolerance must be less than 1.")

    minimum_from_final_activity = (
        -math.log1p(-target_final_nonempty_probability)
        / retention_factor ** (num_stages - 1)
    )
    # The final non-final stage must have bar_mu > m >= 1.
    minimum_from_conditioning = retention_factor ** (-(num_stages - 2))
    search_lower = max(
        minimum_from_final_activity,
        math.nextafter(minimum_from_conditioning, math.inf),
    )
    if search_lower > maximum_initial_expected_num_trials:
        raise ValueError(
            "No initial breadth can satisfy the final-activity and "
            "conditioning constraints within the configured maximum."
        )

    first_retained_count = max(
        math.floor(retention_factor * search_lower),
        1,
    )
    maximum_retained_count = max(
        math.floor(
            retention_factor * maximum_initial_expected_num_trials
        ),
        1,
    )

    def coverage_for(
        expected_num_trials: float,
        retained_count: int,
    ) -> tuple[float, PoissonDistribution]:
        distribution = PoissonDistribution.from_conditional_mean(
            m=retained_count,
            target_mean=expected_num_trials,
        )
        coverage = conditioned_coverage_probability(
            underlying_poisson_rate=distribution.mu,
            retained_count=retained_count,
            num_configurations=num_configurations,
            coverage_criterion=coverage_criterion,
            num_good_configurations=num_good_configurations,
        )
        return coverage, distribution

    for retained_count in range(
        first_retained_count,
        maximum_retained_count + 1,
    ):
        if retained_count == 1:
            interval_lower = search_lower
            interval_upper_boundary = 2.0 / retention_factor
        else:
            interval_lower = max(
                search_lower,
                retained_count / retention_factor,
            )
            interval_upper_boundary = (
                retained_count + 1
            ) / retention_factor
        interval_lower = max(
            interval_lower,
            math.nextafter(float(retained_count), math.inf),
        )
        interval_upper = min(
            maximum_initial_expected_num_trials,
            math.nextafter(interval_upper_boundary, -math.inf),
        )
        if interval_lower > interval_upper:
            continue

        lower_coverage, lower_distribution = coverage_for(
            interval_lower,
            retained_count,
        )
        coverage_tolerance = relative_tolerance * max(
            target_initial_good_coverage,
            1.0,
        )
        if lower_coverage >= (
            target_initial_good_coverage - coverage_tolerance
        ):
            solution_value = interval_lower
            solution_coverage = lower_coverage
            solution_distribution = lower_distribution
        else:
            upper_coverage, _ = coverage_for(
                interval_upper,
                retained_count,
            )
            if upper_coverage < (
                target_initial_good_coverage - coverage_tolerance
            ):
                continue

            def objective(expected_num_trials: float) -> float:
                coverage, _ = coverage_for(
                    expected_num_trials,
                    retained_count,
                )
                return coverage - target_initial_good_coverage

            solution_value = float(
                brentq(
                    objective,
                    interval_lower,
                    interval_upper,
                    xtol=1e-12,
                    rtol=relative_tolerance,
                    maxiter=300,
                )
            )
            solution_coverage, solution_distribution = coverage_for(
                solution_value,
                retained_count,
            )

        final_expected_num_trials = (
            solution_value * retention_factor ** (num_stages - 1)
        )
        final_nonempty_probability = -math.expm1(
            -final_expected_num_trials
        )
        if final_nonempty_probability + coverage_tolerance < (
            target_final_nonempty_probability
        ):
            # This should be excluded by search_lower, but keep the result
            # safe against floating-point movement at the boundary.
            continue
        return InitialBreadthSolution(
            coverage_criterion=coverage_criterion,
            initial_expected_num_trials=float(solution_value),
            stage_1_retained_count=retained_count,
            stage_1_underlying_poisson_rate=solution_distribution.mu,
            initial_coverage_probability=float(solution_coverage),
            final_expected_num_trials=float(final_expected_num_trials),
            final_nonempty_probability=float(final_nonempty_probability),
        )

    raise ValueError(
        "No initial breadth satisfies the requested coverage within "
        "maximum_initial_expected_num_trials."
    )


def evaluate_schedule_constraints(
    schedule: GeometricSchedule,
    *,
    num_configurations: int,
    num_good_configurations: int | None,
    target_initial_good_coverage: float,
    target_final_nonempty_probability: float,
    expected_compute_budget: float,
    coverage_criterion: str = "at_least_one_good",
    relative_tolerance: float = 1e-10,
) -> ScheduleConstraintResult:
    """Evaluate coverage, final activity, and expected-compute constraints."""
    if not isinstance(schedule, GeometricSchedule):
        raise TypeError("schedule must be a GeometricSchedule.")
    coverage_criterion = _normalize_coverage_criterion(coverage_criterion)
    num_configurations, num_good_configurations = (
        _validate_configuration_counts(
            num_configurations=num_configurations,
            num_good_configurations=num_good_configurations,
            coverage_criterion=coverage_criterion,
        )
    )
    target_initial_good_coverage = _validate_open_probability(
        target_initial_good_coverage,
        "target_initial_good_coverage",
    )
    target_final_nonempty_probability = _validate_open_probability(
        target_final_nonempty_probability,
        "target_final_nonempty_probability",
    )
    expected_compute_budget = validate_positive_float(
        expected_compute_budget,
        "expected_compute_budget",
    )
    relative_tolerance = validate_positive_float(
        relative_tolerance,
        "relative_tolerance",
    )
    if relative_tolerance >= 1.0:
        raise ValueError("relative_tolerance must be less than 1.")

    first_stage = schedule.stages[0]
    final_stage = schedule.stages[-1]
    if first_stage.conditioning_probability is None:
        initial_coverage = unconditioned_coverage_probability(
            underlying_poisson_rate=first_stage.underlying_poisson_rate,
            num_configurations=num_configurations,
            coverage_criterion=coverage_criterion,
            num_good_configurations=num_good_configurations,
        )
    else:
        initial_coverage = conditioned_coverage_probability(
            underlying_poisson_rate=(
                first_stage.underlying_poisson_rate
            ),
            retained_count=first_stage.retained_count,
            num_configurations=num_configurations,
            coverage_criterion=coverage_criterion,
            num_good_configurations=num_good_configurations,
        )
    final_nonempty_probability = 1.0 - final_stage.probability_k_zero

    coverage_margin = initial_coverage - target_initial_good_coverage
    final_margin = (
        final_nonempty_probability
        - target_final_nonempty_probability
    )
    compute_margin = expected_compute_budget - schedule.expected_compute
    coverage_tolerance = relative_tolerance * max(
        target_initial_good_coverage,
        1.0,
    )
    final_tolerance = relative_tolerance * max(
        target_final_nonempty_probability,
        1.0,
    )
    compute_tolerance = relative_tolerance * max(
        expected_compute_budget,
        1.0,
    )

    rejection_reasons = []
    if coverage_margin < -coverage_tolerance:
        rejection_reasons.append(
            "initial_coverage_below_target"
        )
    if final_margin < -final_tolerance:
        rejection_reasons.append(
            "final_nonempty_probability_below_target"
        )
    if compute_margin < -compute_tolerance:
        rejection_reasons.append("expected_compute_exceeds_budget")

    return ScheduleConstraintResult(
        coverage_criterion=coverage_criterion,
        is_feasible=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
        expected_compute=schedule.expected_compute,
        expected_compute_budget=float(expected_compute_budget),
        expected_compute_margin=float(compute_margin),
        initial_coverage_probability=float(initial_coverage),
        target_initial_coverage=float(
            target_initial_good_coverage
        ),
        initial_coverage_margin=float(coverage_margin),
        final_nonempty_probability=float(final_nonempty_probability),
        target_final_nonempty_probability=float(
            target_final_nonempty_probability
        ),
        final_nonempty_probability_margin=float(final_margin),
    )


def enumerate_geometric_schedule_candidates(
    *,
    num_stages_values: Iterable[int],
    retention_factors: Iterable[float],
    minimum_resource: int,
    required_final_resource: int,
    num_configurations: int,
    num_good_configurations: int | None,
    target_initial_good_coverage: float,
    target_final_nonempty_probability: float,
    expected_compute_budget: float,
    coverage_criterion: str = "at_least_one_good",
    maximum_initial_expected_num_trials: float = 1_000_000.0,
    survivor_rounding: str = "floor",
    resource_rounding: str = "nearest",
    relative_tolerance: float = 1e-10,
) -> GeometricScheduleSearchResult:
    """Enumerate and evaluate the non-private ``(L, q)`` search space."""
    num_stages_values = tuple(num_stages_values)
    retention_factors = tuple(retention_factors)
    if not num_stages_values:
        raise ValueError("num_stages_values must not be empty.")
    coverage_criterion = _normalize_coverage_criterion(coverage_criterion)
    survivor_rounding = _validate_rounding_mode(
        survivor_rounding,
        "survivor_rounding",
    )
    if survivor_rounding != "floor":
        raise ValueError(
            "The minimum-breadth candidate search currently requires "
            "survivor_rounding='floor'. Manual schedule construction still "
            "supports the other rounding modes."
        )

    normalized_num_stages = tuple(
        sorted(
            {
                validate_positive_integer(value, "num_stages")
                for value in num_stages_values
            }
        )
    )
    if normalized_num_stages[0] < 1:
        raise ValueError("Every candidate must have at least one stage.")
    if any(value >= 2 for value in normalized_num_stages) and not (
        retention_factors
    ):
        raise ValueError(
            "retention_factors must not be empty when searching L >= 2."
        )
    normalized_retention_factors = tuple(
        sorted({_validate_retention_factor(value) for value in retention_factors})
    )

    candidates: list[GeometricScheduleCandidate] = []
    for num_stages in normalized_num_stages:
        stage_retention_factors: tuple[float | None, ...] = (
            (None,)
            if num_stages == 1
            else normalized_retention_factors
        )
        for retention_factor in stage_retention_factors:
            try:
                if num_stages == 1:
                    breadth = find_minimum_feasible_papernot_breadth(
                        num_configurations=num_configurations,
                        num_good_configurations=num_good_configurations,
                        coverage_criterion=coverage_criterion,
                        target_initial_coverage=(
                            target_initial_good_coverage
                        ),
                        target_final_nonempty_probability=(
                            target_final_nonempty_probability
                        ),
                        maximum_initial_expected_num_trials=(
                            maximum_initial_expected_num_trials
                        ),
                    )
                else:
                    if retention_factor is None:
                        raise RuntimeError(
                            "A multi-stage candidate has no retention factor."
                        )
                    breadth = find_minimum_feasible_initial_breadth(
                        retention_factor=retention_factor,
                        num_stages=num_stages,
                        num_configurations=num_configurations,
                        num_good_configurations=num_good_configurations,
                        target_initial_good_coverage=(
                            target_initial_good_coverage
                        ),
                        target_final_nonempty_probability=(
                            target_final_nonempty_probability
                        ),
                        coverage_criterion=coverage_criterion,
                        maximum_initial_expected_num_trials=(
                            maximum_initial_expected_num_trials
                        ),
                        relative_tolerance=relative_tolerance,
                    )
            except (ArithmeticError, ValueError) as error:
                candidates.append(
                    GeometricScheduleCandidate(
                        num_stages=num_stages,
                        retention_factor=retention_factor,
                        schedule=None,
                        constraints=None,
                        rejection_reasons=(
                            "initial_breadth_not_found",
                        ),
                        diagnostic=str(error),
                    )
                )
                continue

            try:
                schedule = build_geometric_schedule(
                    initial_expected_num_trials=(
                        breadth.initial_expected_num_trials
                    ),
                    retention_factor=retention_factor,
                    num_stages=num_stages,
                    minimum_resource=minimum_resource,
                    required_final_resource=required_final_resource,
                    survivor_rounding=survivor_rounding,
                    resource_rounding=resource_rounding,
                )
            except (ArithmeticError, ValueError) as error:
                candidates.append(
                    GeometricScheduleCandidate(
                        num_stages=num_stages,
                        retention_factor=retention_factor,
                        schedule=None,
                        constraints=None,
                        rejection_reasons=(
                            "schedule_construction_failed",
                        ),
                        diagnostic=str(error),
                    )
                )
                continue

            constraints = evaluate_schedule_constraints(
                schedule,
                num_configurations=num_configurations,
                num_good_configurations=num_good_configurations,
                target_initial_good_coverage=(
                    target_initial_good_coverage
                ),
                target_final_nonempty_probability=(
                    target_final_nonempty_probability
                ),
                expected_compute_budget=expected_compute_budget,
                coverage_criterion=coverage_criterion,
                relative_tolerance=relative_tolerance,
            )
            candidates.append(
                GeometricScheduleCandidate(
                    num_stages=num_stages,
                    retention_factor=retention_factor,
                    schedule=schedule,
                    constraints=constraints,
                    rejection_reasons=constraints.rejection_reasons,
                )
            )

    return GeometricScheduleSearchResult(candidates=tuple(candidates))


def evaluate_schedule_privacy(
    schedule: GeometricSchedule,
    *,
    sampling_rate: float,
    noise_multiplier: float,
    orders: Iterable[float],
    delta: float,
) -> SchedulePrivacyEvaluation:
    """Evaluate one fixed schedule with a common Stage-wise ``sigma``.

    Each base curve accounts only for that stage's incremental number of
    DP-SGD updates.  The selection accountant then adds every stage's RDP
    guarantee on the common order grid before the single conversion to
    approximate DP.
    """
    if not isinstance(schedule, GeometricSchedule):
        raise TypeError("schedule must be a GeometricSchedule.")
    sampling_rate = float(sampling_rate)
    if not math.isfinite(sampling_rate) or not 0.0 < sampling_rate <= 1.0:
        raise ValueError(
            "sampling_rate must be finite and satisfy 0 < rate <= 1."
        )
    noise_multiplier = validate_positive_float(
        noise_multiplier,
        "noise_multiplier",
    )
    delta = _validate_open_probability(delta, "delta")
    normalized_orders = normalize_requested_integer_orders(tuple(orders))
    float_orders = normalized_orders.astype(float)

    # DP-SGD RDP is linear in the number of updates.  Computing the
    # numerical one-step curve once and scaling it is exactly equivalent to
    # invoking the accountant independently for every incremental segment,
    # while avoiding L repeated dense-order accounting calls per sigma.
    one_update_curve = compute_dpsgd_rdp(
        config={
            "num_rounds": 1,
            "data_sampling_rate": sampling_rate,
            "sigma_gaussian": noise_multiplier,
        },
        orders=float_orders,
    )
    stage_base_rdp_curves = tuple(
        scale_rdp_curve(one_update_curve, stage.incremental_resource)
        for stage in schedule.stages
    )
    n_stage_result = compute_n_stage_rdp_poisson(
        stage_base_rdp_curves,
        retained_counts=[
            stage.retained_count for stage in schedule.stages
        ],
        expected_num_trials=[
            stage.expected_num_trials for stage in schedule.stages
        ],
    )
    approximate_dp = convert_rdp_to_approx_dp(
        n_stage_result.rdp_curve,
        delta=delta,
    )
    return SchedulePrivacyEvaluation(
        noise_multiplier=noise_multiplier,
        stage_base_rdp_curves=stage_base_rdp_curves,
        n_stage_result=n_stage_result,
        approximate_dp=approximate_dp,
    )


def calibrate_schedule_noise_multiplier(
    schedule: GeometricSchedule,
    *,
    sampling_rate: float,
    target_epsilon: float,
    delta: float,
    orders: Iterable[float],
    initial_sigma: float,
    minimum_sigma: float,
    maximum_sigma: float,
    relative_sigma_tolerance: float = 1e-6,
    max_iterations: int = 80,
    reject_renyi_order_boundary: bool = True,
) -> ScheduleSigmaCalibrationResult:
    """Find the smallest configured common sigma meeting ``target_epsilon``.

    Search is performed geometrically because privacy changes over multiple
    scales of ``sigma``.  The lower endpoint is privacy-infeasible and the
    upper endpoint is feasible.  If ``minimum_sigma`` is already feasible,
    it is returned explicitly as the constrained optimum.
    """
    if not isinstance(schedule, GeometricSchedule):
        raise TypeError("schedule must be a GeometricSchedule.")
    target_epsilon = validate_positive_float(
        target_epsilon,
        "target_epsilon",
    )
    initial_sigma = validate_positive_float(initial_sigma, "initial_sigma")
    minimum_sigma = validate_positive_float(minimum_sigma, "minimum_sigma")
    maximum_sigma = validate_positive_float(maximum_sigma, "maximum_sigma")
    if minimum_sigma >= maximum_sigma:
        raise ValueError("minimum_sigma must be smaller than maximum_sigma.")
    if not minimum_sigma <= initial_sigma <= maximum_sigma:
        raise ValueError(
            "initial_sigma must lie in [minimum_sigma, maximum_sigma]."
        )
    relative_sigma_tolerance = validate_positive_float(
        relative_sigma_tolerance,
        "relative_sigma_tolerance",
    )
    if relative_sigma_tolerance >= 1.0:
        raise ValueError("relative_sigma_tolerance must be less than 1.")
    if not isinstance(max_iterations, Integral) or isinstance(
        max_iterations,
        bool,
    ):
        raise TypeError("max_iterations must be an integer.")
    max_iterations = int(max_iterations)
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    # Materialize once because callers may provide a generator.
    normalized_orders = normalize_requested_integer_orders(tuple(orders))
    float_orders = normalized_orders.astype(float)
    evaluations: dict[float, SchedulePrivacyEvaluation] = {}

    def evaluate(sigma: float) -> SchedulePrivacyEvaluation:
        sigma = float(sigma)
        result = evaluations.get(sigma)
        if result is None:
            result = evaluate_schedule_privacy(
                schedule,
                sampling_rate=sampling_rate,
                noise_multiplier=sigma,
                orders=float_orders,
                delta=delta,
            )
            evaluations[sigma] = result
        return result

    minimum_result = evaluate(minimum_sigma)
    epsilon_tolerance = relative_sigma_tolerance * max(
        target_epsilon,
        1.0,
    )
    if minimum_result.approximate_dp.epsilon <= target_epsilon:
        selected_result = minimum_result
        lower_sigma = None
        lower_epsilon = None
        upper_sigma = minimum_sigma
        iterations = 0
        at_minimum_sigma = True
    else:
        maximum_result = evaluate(maximum_sigma)
        if (
            minimum_result.approximate_dp.epsilon
            + epsilon_tolerance
            < maximum_result.approximate_dp.epsilon
        ):
            raise SchedulePrivacyCalibrationError(
                "privacy_not_monotone_in_sigma",
                "Privacy epsilon unexpectedly increased as sigma increased "
                "across the configured search interval.",
            )
        if maximum_result.approximate_dp.epsilon > target_epsilon:
            raise SchedulePrivacyCalibrationError(
                "sigma_target_not_bracketed",
                "maximum_sigma does not satisfy the target privacy budget; "
                f"epsilon({maximum_sigma:g})="
                f"{maximum_result.approximate_dp.epsilon:g} exceeds "
                f"target_epsilon={target_epsilon:g}.",
            )

        lower_sigma = minimum_sigma
        lower_result = minimum_result
        upper_sigma = maximum_sigma
        upper_result = maximum_result
        if minimum_sigma < initial_sigma < maximum_sigma:
            initial_result = evaluate(initial_sigma)
            if initial_result.approximate_dp.epsilon > target_epsilon:
                lower_sigma = initial_sigma
                lower_result = initial_result
            else:
                upper_sigma = initial_sigma
                upper_result = initial_result

        iterations = 0
        for iterations in range(1, max_iterations + 1):
            relative_width = (
                upper_sigma - lower_sigma
            ) / upper_sigma
            if relative_width <= relative_sigma_tolerance:
                break
            midpoint_sigma = math.sqrt(lower_sigma * upper_sigma)
            midpoint_result = evaluate(midpoint_sigma)
            if midpoint_result.approximate_dp.epsilon > target_epsilon:
                lower_sigma = midpoint_sigma
                lower_result = midpoint_result
            else:
                upper_sigma = midpoint_sigma
                upper_result = midpoint_result
        else:
            raise SchedulePrivacyCalibrationError(
                "sigma_calibration_did_not_converge",
                "Common-sigma calibration did not converge within "
                f"max_iterations={max_iterations}.",
            )

        selected_result = upper_result
        lower_epsilon = lower_result.approximate_dp.epsilon
        at_minimum_sigma = False

    approximate_dp = selected_result.approximate_dp
    if approximate_dp.epsilon > target_epsilon:
        raise SchedulePrivacyCalibrationError(
            "sigma_calibration_returned_infeasible_result",
            "The calibrated upper endpoint does not satisfy target_epsilon.",
        )
    if reject_renyi_order_boundary and (
        approximate_dp.is_at_min_order or approximate_dp.is_at_max_order
    ):
        boundary = (
            "minimum"
            if approximate_dp.is_at_min_order
            else "maximum"
        )
        raise SchedulePrivacyCalibrationError(
            "renyi_order_boundary",
            "The calibrated privacy result uses the "
            f"{boundary} configured Renyi order "
            f"({approximate_dp.best_order:g}); expand the order grid.",
        )

    return ScheduleSigmaCalibrationResult(
        target_epsilon=target_epsilon,
        achieved_epsilon=approximate_dp.epsilon,
        delta=approximate_dp.delta,
        noise_multiplier=selected_result.noise_multiplier,
        best_renyi_order=approximate_dp.best_order,
        relative_sigma_tolerance=relative_sigma_tolerance,
        bisection_iterations=iterations,
        accountant_evaluations=len(evaluations),
        at_minimum_sigma=at_minimum_sigma,
        infeasible_lower_sigma=lower_sigma,
        infeasible_lower_epsilon=lower_epsilon,
        feasible_upper_sigma=selected_result.noise_multiplier,
        feasible_upper_epsilon=approximate_dp.epsilon,
        privacy_evaluation=selected_result,
    )


def calibrate_schedule_candidates(
    structural_search: GeometricScheduleSearchResult,
    *,
    sampling_rate: float,
    target_epsilon: float,
    delta: float,
    orders: Iterable[float],
    initial_sigma: float,
    minimum_sigma: float,
    maximum_sigma: float,
    relative_sigma_tolerance: float = 1e-6,
    max_iterations: int = 80,
    reject_renyi_order_boundary: bool = True,
) -> PrivacyCalibratedScheduleSearchResult:
    """Calibrate every structurally feasible candidate without dropping any."""
    if not isinstance(structural_search, GeometricScheduleSearchResult):
        raise TypeError(
            "structural_search must be a GeometricScheduleSearchResult."
        )
    normalized_orders = normalize_requested_integer_orders(tuple(orders))
    float_orders = normalized_orders.astype(float)
    calibrated_candidates = []

    for structural_candidate in structural_search.candidates:
        if not structural_candidate.is_feasible:
            calibrated_candidates.append(
                PrivacyCalibratedScheduleCandidate(
                    structural_candidate=structural_candidate,
                    calibration=None,
                    rejection_reasons=(
                        structural_candidate.rejection_reasons
                    ),
                    diagnostic=structural_candidate.diagnostic,
                )
            )
            continue

        schedule = structural_candidate.schedule
        if schedule is None:
            raise RuntimeError(
                "A structurally feasible candidate has no schedule."
            )
        try:
            calibration = calibrate_schedule_noise_multiplier(
                schedule,
                sampling_rate=sampling_rate,
                target_epsilon=target_epsilon,
                delta=delta,
                orders=float_orders,
                initial_sigma=initial_sigma,
                minimum_sigma=minimum_sigma,
                maximum_sigma=maximum_sigma,
                relative_sigma_tolerance=relative_sigma_tolerance,
                max_iterations=max_iterations,
                reject_renyi_order_boundary=(
                    reject_renyi_order_boundary
                ),
            )
        except SchedulePrivacyCalibrationError as error:
            calibrated_candidates.append(
                PrivacyCalibratedScheduleCandidate(
                    structural_candidate=structural_candidate,
                    calibration=None,
                    rejection_reasons=(error.reason,),
                    diagnostic=str(error),
                )
            )
            continue

        calibrated_candidates.append(
            PrivacyCalibratedScheduleCandidate(
                structural_candidate=structural_candidate,
                calibration=calibration,
                rejection_reasons=(),
            )
        )

    return PrivacyCalibratedScheduleSearchResult(
        candidates=tuple(calibrated_candidates)
    )
