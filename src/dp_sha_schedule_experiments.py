"""Hydra orchestration for fixed expected-budget DP-SHA planning.

The mathematical schedule construction and privacy calibration live in
``privacy_accounting.schedule_optimization``.  This file only validates an
experiment configuration, runs the deterministic grid search, selects and
persists plans, and produces diagnostic plots.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Iterable

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from privacy_accounting.schedule_optimization import (
    GeometricScheduleCandidate,
    GeometricScheduleSearchResult,
    PrivacyCalibratedScheduleCandidate,
    PrivacyCalibratedScheduleSearchResult,
    calibrate_schedule_candidates,
    enumerate_geometric_schedule_candidates,
)


_SCHEMA_VERSION = 2
_RUNNER_NAME = "optimize_fixed_expected_budget_dp_sha"

_CANDIDATE_FIELDS = (
    "candidate_id",
    "candidate_index",
    "mechanism",
    "num_stages",
    "retention_factor_q",
    "structurally_feasible",
    "privacy_feasible",
    "is_pareto_optimal",
    "is_selected",
    "status",
    "rejection_reasons",
    "diagnostic",
    "initial_expected_workload_w_1",
    "final_expected_workload_w_L",
    "stage_1_retained_count_m_1",
    "resource_growth_factor_rho",
    "minimum_resource_r_1",
    "required_final_resource_r_L",
    "expected_compute",
    "expected_compute_budget",
    "expected_compute_margin",
    "coverage_criterion",
    "initial_coverage_probability",
    "target_initial_coverage",
    "initial_coverage_margin",
    "final_nonempty_probability",
    "target_final_nonempty_probability",
    "final_nonempty_probability_margin",
    "target_epsilon",
    "achieved_epsilon",
    "delta",
    "calibrated_common_noise_multiplier_sigma",
    "best_renyi_order",
    "at_minimum_sigma",
    "bisection_iterations",
    "accountant_evaluations",
    "calibration_execution_mode",
    "calibration_worker_pid",
    "calibration_wall_time_seconds",
    "calibration_started_at_utc",
    "calibration_finished_at_utc",
    "infeasible_lower_sigma",
    "infeasible_lower_epsilon",
    "feasible_upper_sigma",
    "feasible_upper_epsilon",
)


@dataclass(frozen=True, slots=True)
class _CandidateCalibrationTask:
    """Pickle-safe input for one complete candidate sigma calibration."""

    candidate_index: int
    structural_candidate: GeometricScheduleCandidate
    sampling_rate: float
    target_epsilon: float
    delta: float
    orders: tuple[float, ...]
    initial_sigma: float
    minimum_sigma: float
    maximum_sigma: float
    relative_sigma_tolerance: float
    max_iterations: int
    reject_renyi_order_boundary: bool
    execution_mode: str


@dataclass(frozen=True, slots=True)
class _CandidateCalibrationTaskResult:
    """One calibrated candidate plus worker execution diagnostics."""

    candidate_index: int
    calibrated_candidate: PrivacyCalibratedScheduleCandidate
    execution_mode: str
    worker_pid: int
    started_at_utc: str
    finished_at_utc: str
    wall_time_seconds: float


@dataclass(frozen=True, slots=True)
class _CalibrationExecutionDiagnostic:
    """Parent-side execution record, including structural rejections."""

    candidate_index: int
    submitted_for_calibration: bool
    execution_mode: str
    worker_pid: int | None
    started_at_utc: str | None
    finished_at_utc: str | None
    wall_time_seconds: float
    status: str


@dataclass(frozen=True, slots=True)
class _CalibratedSearchExecution:
    """Deterministic calibrated search and aggregate timing metadata."""

    search_result: PrivacyCalibratedScheduleSearchResult
    diagnostics: tuple[_CalibrationExecutionDiagnostic, ...]
    requested_num_workers: str
    resolved_num_workers: int
    worker_resolution_source: str
    multiprocessing_start_method: str
    calibration_wall_time_seconds: float
    summed_candidate_wall_time_seconds: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_single_thread_worker_environment() -> None:
    """Prevent every process from creating its own BLAS thread pool."""
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"


def _run_candidate_calibration_task(
    task: _CandidateCalibrationTask,
) -> _CandidateCalibrationTaskResult:
    """Calibrate one candidate in either the parent or a worker process."""
    if task.execution_mode == "process_pool":
        _set_single_thread_worker_environment()
    started_at_utc = _utc_now()
    start_time = time.perf_counter()
    single_candidate_search = GeometricScheduleSearchResult(
        candidates=(task.structural_candidate,)
    )
    calibrated_candidate = calibrate_schedule_candidates(
        single_candidate_search,
        sampling_rate=task.sampling_rate,
        target_epsilon=task.target_epsilon,
        delta=task.delta,
        orders=task.orders,
        initial_sigma=task.initial_sigma,
        minimum_sigma=task.minimum_sigma,
        maximum_sigma=task.maximum_sigma,
        relative_sigma_tolerance=task.relative_sigma_tolerance,
        max_iterations=task.max_iterations,
        reject_renyi_order_boundary=task.reject_renyi_order_boundary,
    ).candidates[0]
    wall_time_seconds = time.perf_counter() - start_time
    return _CandidateCalibrationTaskResult(
        candidate_index=task.candidate_index,
        calibrated_candidate=calibrated_candidate,
        execution_mode=task.execution_mode,
        worker_pid=os.getpid(),
        started_at_utc=started_at_utc,
        finished_at_utc=_utc_now(),
        wall_time_seconds=float(wall_time_seconds),
    )

_STAGE_FIELDS = (
    "candidate_id",
    "candidate_index",
    "mechanism",
    "privacy_feasible",
    "is_selected",
    "num_stages",
    "retention_factor_q",
    "stage",
    "is_final_stage",
    "count_semantics",
    "expected_num_trials",
    "retained_count_m_l",
    "underlying_poisson_rate_theta_l",
    "conditioning_probability",
    "probability_k_zero",
    "planned_retention_ratio_m_l_over_expected_trials",
    "cumulative_resource_r_l",
    "incremental_resource_delta_r_l",
    "stage_expected_compute",
    "common_noise_multiplier_sigma",
    "total_best_renyi_order",
    "base_rdp_at_total_best_order",
    "selected_stage_rdp_at_total_best_order",
    "raw_selected_stage_rdp_at_total_best_order",
    "hat_epsilon_at_total_best_order",
    "hat_delta_at_total_best_order",
    "best_auxiliary_order_at_total_best_order",
    "log_expected_binomial",
)


def _validate_positive_float(value: object, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return value


def _validate_open_probability(value: object, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be finite and lie strictly in (0, 1).")
    return value


def _validate_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool.")
    integer = int(value)
    if float(value) != integer or integer <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _resolve_num_workers(
    configured_value: object,
    *,
    num_feasible_candidates: int,
) -> tuple[str, int, str]:
    """Resolve ``auto`` safely inside and outside a Slurm allocation."""
    requested = str(configured_value).strip().lower()
    slurm_cpu_text = os.environ.get("SLURM_CPUS_PER_TASK")
    slurm_cpu_limit = None
    if slurm_cpu_text is not None:
        slurm_cpu_limit = _validate_positive_integer(
            slurm_cpu_text,
            "SLURM_CPUS_PER_TASK",
        )

    if requested == "auto":
        if slurm_cpu_limit is None:
            resolved = 1
            source = "auto_local_serial_default"
        else:
            resolved = slurm_cpu_limit
            source = "auto_slurm_cpus_per_task"
    else:
        resolved = _validate_positive_integer(
            configured_value,
            "search.num_workers",
        )
        source = "explicit_configuration"
        if slurm_cpu_limit is not None and resolved > slurm_cpu_limit:
            resolved = slurm_cpu_limit
            source = "explicit_capped_by_slurm_cpus_per_task"

    if num_feasible_candidates > 0:
        resolved = min(resolved, num_feasible_candidates)
    else:
        resolved = 1
    return requested, resolved, source


def _calibrate_schedule_candidates_with_workers(
    structural_search: GeometricScheduleSearchResult,
    *,
    sampling_rate: float,
    target_epsilon: float,
    delta: float,
    orders: tuple[float, ...],
    initial_sigma: float,
    minimum_sigma: float,
    maximum_sigma: float,
    relative_sigma_tolerance: float,
    max_iterations: int,
    reject_renyi_order_boundary: bool,
    configured_num_workers: object,
    multiprocessing_start_method: str,
) -> _CalibratedSearchExecution:
    """Calibrate independent candidates concurrently and restore grid order."""
    if not isinstance(structural_search, GeometricScheduleSearchResult):
        raise TypeError(
            "structural_search must be a GeometricScheduleSearchResult."
        )
    start_method = str(multiprocessing_start_method).strip().lower()
    available_start_methods = multiprocessing.get_all_start_methods()
    if start_method not in available_start_methods:
        available = ", ".join(available_start_methods)
        raise ValueError(
            "search.multiprocessing_start_method must be supported by this "
            f"platform; available methods: {available}."
        )
    requested_workers, resolved_workers, resolution_source = (
        _resolve_num_workers(
            configured_num_workers,
            num_feasible_candidates=len(
                structural_search.feasible_candidates
            ),
        )
    )
    execution_mode = "serial" if resolved_workers == 1 else "process_pool"
    calibrated_candidates: list[
        PrivacyCalibratedScheduleCandidate | None
    ] = [None] * len(structural_search.candidates)
    execution_diagnostics: list[
        _CalibrationExecutionDiagnostic | None
    ] = [None] * len(structural_search.candidates)
    tasks = []

    for candidate_index, structural_candidate in enumerate(
        structural_search.candidates
    ):
        if not structural_candidate.is_feasible:
            calibrated_candidates[candidate_index] = (
                PrivacyCalibratedScheduleCandidate(
                    structural_candidate=structural_candidate,
                    calibration=None,
                    rejection_reasons=(
                        structural_candidate.rejection_reasons
                    ),
                    diagnostic=structural_candidate.diagnostic,
                )
            )
            execution_diagnostics[candidate_index] = (
                _CalibrationExecutionDiagnostic(
                    candidate_index=candidate_index,
                    submitted_for_calibration=False,
                    execution_mode="not_submitted",
                    worker_pid=None,
                    started_at_utc=None,
                    finished_at_utc=None,
                    wall_time_seconds=0.0,
                    status="structural_rejection",
                )
            )
            continue
        tasks.append(
            _CandidateCalibrationTask(
                candidate_index=candidate_index,
                structural_candidate=structural_candidate,
                sampling_rate=sampling_rate,
                target_epsilon=target_epsilon,
                delta=delta,
                orders=orders,
                initial_sigma=initial_sigma,
                minimum_sigma=minimum_sigma,
                maximum_sigma=maximum_sigma,
                relative_sigma_tolerance=relative_sigma_tolerance,
                max_iterations=max_iterations,
                reject_renyi_order_boundary=reject_renyi_order_boundary,
                execution_mode=execution_mode,
            )
        )

    def record_result(
        result: _CandidateCalibrationTaskResult,
        completed_count: int,
    ) -> None:
        candidate_index = result.candidate_index
        calibrated_candidate = result.calibrated_candidate
        calibrated_candidates[candidate_index] = calibrated_candidate
        status = (
            "privacy_feasible"
            if calibrated_candidate.is_feasible
            else "privacy_rejection"
        )
        execution_diagnostics[candidate_index] = (
            _CalibrationExecutionDiagnostic(
                candidate_index=candidate_index,
                submitted_for_calibration=True,
                execution_mode=result.execution_mode,
                worker_pid=result.worker_pid,
                started_at_utc=result.started_at_utc,
                finished_at_utc=result.finished_at_utc,
                wall_time_seconds=result.wall_time_seconds,
                status=status,
            )
        )
        structural = calibrated_candidate.structural_candidate
        q_text = (
            "n/a"
            if structural.retention_factor is None
            else format(structural.retention_factor, ".6g")
        )
        print(
            "Candidate calibration "
            f"{completed_count}/{len(tasks)}: "
            f"L={structural.num_stages}, q={q_text}, "
            f"status={status}, worker_pid={result.worker_pid}, "
            f"wall_time={result.wall_time_seconds:.3f}s",
            flush=True,
        )

    if resolved_workers > 1:
        _set_single_thread_worker_environment()
    calibration_start = time.perf_counter()
    if resolved_workers == 1:
        for completed_count, task in enumerate(tasks, start=1):
            record_result(
                _run_candidate_calibration_task(task),
                completed_count,
            )
    elif tasks:
        process_context = multiprocessing.get_context(start_method)
        with ProcessPoolExecutor(
            max_workers=resolved_workers,
            mp_context=process_context,
        ) as executor:
            future_to_task = {
                executor.submit(_run_candidate_calibration_task, task): task
                for task in tasks
            }
            completed_count = 0
            try:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        for pending_future in future_to_task:
                            pending_future.cancel()
                        structural = task.structural_candidate
                        raise RuntimeError(
                            "Parallel privacy calibration failed for "
                            f"candidate index {task.candidate_index + 1}, "
                            f"L={structural.num_stages}, "
                            f"q={structural.retention_factor}."
                        ) from error
                    completed_count += 1
                    record_result(result, completed_count)
            finally:
                for future in future_to_task:
                    future.cancel()
    calibration_wall_time_seconds = time.perf_counter() - calibration_start

    if any(candidate is None for candidate in calibrated_candidates):
        raise RuntimeError(
            "Candidate calibration finished with missing candidate results."
        )
    if any(diagnostic is None for diagnostic in execution_diagnostics):
        raise RuntimeError(
            "Candidate calibration finished with missing timing diagnostics."
        )
    final_candidates = tuple(
        candidate
        for candidate in calibrated_candidates
        if candidate is not None
    )
    final_diagnostics = tuple(
        diagnostic
        for diagnostic in execution_diagnostics
        if diagnostic is not None
    )
    summed_candidate_wall_time_seconds = math.fsum(
        diagnostic.wall_time_seconds
        for diagnostic in final_diagnostics
        if diagnostic.submitted_for_calibration
    )
    return _CalibratedSearchExecution(
        search_result=PrivacyCalibratedScheduleSearchResult(
            candidates=final_candidates
        ),
        diagnostics=final_diagnostics,
        requested_num_workers=requested_workers,
        resolved_num_workers=resolved_workers,
        worker_resolution_source=resolution_source,
        multiprocessing_start_method=start_method,
        calibration_wall_time_seconds=float(calibration_wall_time_seconds),
        summed_candidate_wall_time_seconds=float(
            summed_candidate_wall_time_seconds
        ),
    )


def _stage_count_values(schedule_config: DictConfig) -> tuple[int, ...]:
    stage_config = schedule_config.num_stages
    if "values" in stage_config:
        values = tuple(
            _validate_positive_integer(value, "schedule.num_stages.values")
            for value in stage_config["values"]
        )
    else:
        minimum = _validate_positive_integer(
            stage_config.minimum,
            "schedule.num_stages.minimum",
        )
        maximum = _validate_positive_integer(
            stage_config.maximum,
            "schedule.num_stages.maximum",
        )
        if maximum < minimum:
            raise ValueError(
                "schedule.num_stages.maximum cannot be below minimum."
            )
        values = tuple(range(minimum, maximum + 1))
    values = tuple(sorted(set(values)))
    if not values or values[0] < 1:
        raise ValueError(
            "Every schedule must contain at least one stage. L=1 is the "
            "Papernot--Steinke Poisson boundary case."
        )
    return values


def _retention_factors(schedule_config: DictConfig) -> tuple[float, ...]:
    values = tuple(
        float(value)
        for value in schedule_config.get("retention_factors", [])
    )
    if not values:
        raise ValueError("schedule.retention_factors must not be empty.")
    for value in values:
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(
                "Every retention factor q must be finite and lie in (0, 1)."
            )
    return tuple(sorted(set(values)))


def _renyi_orders(privacy_config: DictConfig) -> tuple[float, ...]:
    minimum = _validate_positive_integer(
        privacy_config.min_renyi_order,
        "privacy.min_renyi_order",
    )
    maximum = _validate_positive_integer(
        privacy_config.max_renyi_order,
        "privacy.max_renyi_order",
    )
    if minimum < 2 or maximum < minimum:
        raise ValueError(
            "The Renyi-order grid must be an integer interval starting at "
            "an order of at least 2."
        )
    return tuple(float(value) for value in range(minimum, maximum + 1))


def _candidate_key(
    candidate: PrivacyCalibratedScheduleCandidate,
) -> tuple[int, float | None]:
    structural = candidate.structural_candidate
    return structural.num_stages, structural.retention_factor


def _candidate_id(
    candidate: PrivacyCalibratedScheduleCandidate,
    candidate_index: int,
) -> str:
    num_stages, retention_factor = _candidate_key(candidate)
    if num_stages == 1:
        return f"candidate_{candidate_index:04d}_L1_papernot_poisson"
    if retention_factor is None:
        raise RuntimeError("A multi-stage candidate has no retention factor.")
    q_text = format(retention_factor, ".12g").replace("-", "m").replace(
        ".", "p"
    )
    return f"candidate_{candidate_index:04d}_L{num_stages}_q{q_text}"


def _mechanism_name(candidate: PrivacyCalibratedScheduleCandidate) -> str:
    return (
        "papernot_steinke_poisson"
        if candidate.structural_candidate.num_stages == 1
        else "dp_sha_poisson"
    )


def select_minimum_sigma_candidate(
    search_result: PrivacyCalibratedScheduleSearchResult,
) -> PrivacyCalibratedScheduleCandidate:
    """Select the privacy-feasible plan with deterministic tie-breaking."""
    feasible = search_result.feasible_candidates
    if not feasible:
        raise ValueError("No privacy-feasible DP-SHA schedule is available.")

    def selection_key(
        candidate: PrivacyCalibratedScheduleCandidate,
    ) -> tuple[float, float, float, int, float]:
        schedule = candidate.structural_candidate.schedule
        calibration = candidate.calibration
        if schedule is None or calibration is None:
            raise RuntimeError("A feasible candidate is missing its result.")
        return (
            calibration.noise_multiplier,
            schedule.expected_compute,
            -schedule.stages[-1].expected_num_trials,
            schedule.num_stages,
            -(
                schedule.retention_factor
                if schedule.retention_factor is not None
                else 0.0
            ),
        )

    return min(feasible, key=selection_key)


def pareto_optimal_candidates(
    search_result: PrivacyCalibratedScheduleSearchResult,
) -> tuple[PrivacyCalibratedScheduleCandidate, ...]:
    """Return the three-objective Pareto frontier in search order.

    Noise and expected compute are minimized; the final expected workload is
    maximized.  This frontier is diagnostic only.  The official plan is
    selected by :func:`select_minimum_sigma_candidate`.
    """
    feasible = search_result.feasible_candidates

    def objectives(
        candidate: PrivacyCalibratedScheduleCandidate,
    ) -> tuple[float, float, float]:
        schedule = candidate.structural_candidate.schedule
        calibration = candidate.calibration
        if schedule is None or calibration is None:
            raise RuntimeError("A feasible candidate is missing its result.")
        return (
            calibration.noise_multiplier,
            schedule.expected_compute,
            schedule.stages[-1].expected_num_trials,
        )

    frontier = []
    for candidate in feasible:
        sigma, compute, final_workload = objectives(candidate)
        dominated = False
        for other in feasible:
            if other is candidate:
                continue
            other_sigma, other_compute, other_final_workload = objectives(other)
            weakly_better = (
                other_sigma <= sigma
                and other_compute <= compute
                and other_final_workload >= final_workload
            )
            strictly_better = (
                other_sigma < sigma
                or other_compute < compute
                or other_final_workload > final_workload
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(frontier)


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: tuple[str, ...]) -> None:
    rows = tuple(rows)
    with path.open(mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    with path.open(mode="w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def _blank_or(value: object | None) -> object:
    return "" if value is None else value


def _candidate_rows(
    search_result: PrivacyCalibratedScheduleSearchResult,
    *,
    pareto_keys: set[tuple[int, float | None]],
    selected_key: tuple[int, float | None] | None,
    execution_diagnostics: tuple[
        _CalibrationExecutionDiagnostic, ...
    ],
) -> list[dict]:
    if len(execution_diagnostics) != len(search_result.candidates):
        raise ValueError(
            "execution_diagnostics must contain one entry per candidate."
        )
    rows = []
    for index, candidate in enumerate(search_result.candidates, start=1):
        execution = execution_diagnostics[index - 1]
        if execution.candidate_index != index - 1:
            raise RuntimeError(
                "Calibration execution diagnostics are not in candidate order."
            )
        structural = candidate.structural_candidate
        schedule = structural.schedule
        constraints = structural.constraints
        calibration = candidate.calibration
        key = _candidate_key(candidate)
        row = {field: "" for field in _CANDIDATE_FIELDS}
        row.update(
            {
                "candidate_id": _candidate_id(candidate, index),
                "candidate_index": index,
                "mechanism": _mechanism_name(candidate),
                "num_stages": structural.num_stages,
                "retention_factor_q": _blank_or(
                    structural.retention_factor
                ),
                "structurally_feasible": structural.is_feasible,
                "privacy_feasible": candidate.is_feasible,
                "is_pareto_optimal": key in pareto_keys,
                "is_selected": key == selected_key,
                "status": (
                    "privacy_feasible"
                    if candidate.is_feasible
                    else "rejected"
                ),
                "rejection_reasons": ";".join(candidate.rejection_reasons),
                "diagnostic": candidate.diagnostic or "",
                "calibration_execution_mode": execution.execution_mode,
                "calibration_worker_pid": _blank_or(execution.worker_pid),
                "calibration_wall_time_seconds": (
                    execution.wall_time_seconds
                ),
                "calibration_started_at_utc": _blank_or(
                    execution.started_at_utc
                ),
                "calibration_finished_at_utc": _blank_or(
                    execution.finished_at_utc
                ),
            }
        )
        if schedule is not None:
            row.update(
                {
                    "initial_expected_workload_w_1": (
                        schedule.initial_expected_num_trials
                    ),
                    "final_expected_workload_w_L": (
                        schedule.stages[-1].expected_num_trials
                    ),
                    "stage_1_retained_count_m_1": (
                        schedule.stages[0].retained_count
                    ),
                    "resource_growth_factor_rho": (
                        schedule.resource_growth_factor
                    ),
                    "minimum_resource_r_1": schedule.minimum_resource,
                    "required_final_resource_r_L": (
                        schedule.required_final_resource
                    ),
                    "expected_compute": schedule.expected_compute,
                }
            )
        if constraints is not None:
            row.update(
                {
                    "expected_compute_budget": (
                        constraints.expected_compute_budget
                    ),
                    "expected_compute_margin": constraints.expected_compute_margin,
                    "coverage_criterion": constraints.coverage_criterion,
                    "initial_coverage_probability": (
                        constraints.initial_coverage_probability
                    ),
                    "target_initial_coverage": (
                        constraints.target_initial_coverage
                    ),
                    "initial_coverage_margin": (
                        constraints.initial_coverage_margin
                    ),
                    "final_nonempty_probability": (
                        constraints.final_nonempty_probability
                    ),
                    "target_final_nonempty_probability": (
                        constraints.target_final_nonempty_probability
                    ),
                    "final_nonempty_probability_margin": (
                        constraints.final_nonempty_probability_margin
                    ),
                }
            )
        if calibration is not None:
            row.update(
                {
                    "target_epsilon": calibration.target_epsilon,
                    "achieved_epsilon": calibration.achieved_epsilon,
                    "delta": calibration.delta,
                    "calibrated_common_noise_multiplier_sigma": (
                        calibration.noise_multiplier
                    ),
                    "best_renyi_order": calibration.best_renyi_order,
                    "at_minimum_sigma": calibration.at_minimum_sigma,
                    "bisection_iterations": calibration.bisection_iterations,
                    "accountant_evaluations": calibration.accountant_evaluations,
                    "infeasible_lower_sigma": _blank_or(
                        calibration.infeasible_lower_sigma
                    ),
                    "infeasible_lower_epsilon": _blank_or(
                        calibration.infeasible_lower_epsilon
                    ),
                    "feasible_upper_sigma": calibration.feasible_upper_sigma,
                    "feasible_upper_epsilon": (
                        calibration.feasible_upper_epsilon
                    ),
                }
            )
        rows.append(row)
    return rows


def _stage_rows(
    search_result: PrivacyCalibratedScheduleSearchResult,
    *,
    selected_key: tuple[int, float | None] | None,
) -> list[dict]:
    rows = []
    for index, candidate in enumerate(search_result.candidates, start=1):
        structural = candidate.structural_candidate
        schedule = structural.schedule
        calibration = candidate.calibration
        if schedule is None:
            continue
        candidate_id = _candidate_id(candidate, index)
        best_index = None
        stage_results = None
        if calibration is not None:
            privacy = calibration.privacy_evaluation
            best_index = privacy.approximate_dp.best_index
            stage_results = privacy.n_stage_result.stage_results

        for stage_index, stage in enumerate(schedule.stages):
            stage_result = (
                None if stage_results is None else stage_results[stage_index]
            )
            is_conditioned = stage.conditioning_probability is not None
            row = {field: "" for field in _STAGE_FIELDS}
            row.update(
                {
                    "candidate_id": candidate_id,
                    "candidate_index": index,
                    "mechanism": _mechanism_name(candidate),
                    "privacy_feasible": candidate.is_feasible,
                    "is_selected": _candidate_key(candidate) == selected_key,
                    "num_stages": schedule.num_stages,
                    "retention_factor_q": _blank_or(
                        schedule.retention_factor
                    ),
                    "stage": stage.stage,
                    "is_final_stage": stage.stage == schedule.num_stages,
                    "count_semantics": stage.count_semantics,
                    "expected_num_trials": stage.expected_num_trials,
                    "retained_count_m_l": stage.retained_count,
                    "underlying_poisson_rate_theta_l": (
                        stage.underlying_poisson_rate
                    ),
                    "conditioning_probability": _blank_or(
                        stage.conditioning_probability
                    ),
                    "probability_k_zero": stage.probability_k_zero,
                    "planned_retention_ratio_m_l_over_expected_trials": (
                        stage.retained_count / stage.expected_num_trials
                    ),
                    "cumulative_resource_r_l": stage.cumulative_resource,
                    "incremental_resource_delta_r_l": (
                        stage.incremental_resource
                    ),
                    "stage_expected_compute": stage.expected_compute,
                }
            )
            if calibration is not None and stage_result is not None:
                if best_index is None:
                    raise RuntimeError("Privacy result has no best order index.")
                row.update(
                    {
                        "common_noise_multiplier_sigma": (
                            calibration.noise_multiplier
                        ),
                        "total_best_renyi_order": calibration.best_renyi_order,
                        "base_rdp_at_total_best_order": (
                            stage_result.base_rdp_curve.epsilons[best_index]
                        ),
                        "selected_stage_rdp_at_total_best_order": (
                            stage_result.rdp_curve.epsilons[best_index]
                        ),
                        "raw_selected_stage_rdp_at_total_best_order": (
                            stage_result.raw_rdp_curve.epsilons[best_index]
                        ),
                        "hat_epsilon_at_total_best_order": (
                            stage_result.hat_epsilons[best_index]
                        ),
                        "hat_delta_at_total_best_order": (
                            stage_result.hat_deltas[best_index]
                        ),
                        "best_auxiliary_order_at_total_best_order": (
                            stage_result.best_auxiliary_orders[best_index]
                        ),
                        "log_expected_binomial": (
                            stage_result.log_expected_binomial
                            if is_conditioned
                            else ""
                        ),
                    }
                )
            rows.append(row)
    return rows


def _selected_schedule_document(
    candidate: PrivacyCalibratedScheduleCandidate,
    *,
    candidate_id: str,
    sampling_rate: float,
) -> dict:
    structural = candidate.structural_candidate
    schedule = structural.schedule
    constraints = structural.constraints
    calibration = candidate.calibration
    if schedule is None or constraints is None or calibration is None:
        raise RuntimeError("The selected candidate is incomplete.")
    is_papernot = schedule.num_stages == 1
    return {
        "schema_version": _SCHEMA_VERSION,
        "mechanism": (
            "papernot_steinke_poisson"
            if is_papernot
            else "fixed_expected_budget_geometric_dp_sha_poisson"
        ),
        "candidate_id": candidate_id,
        "selection_rule": {
            "primary": "minimum_calibrated_common_noise_multiplier_sigma",
            "tie_breaking": [
                "minimum_expected_compute",
                "maximum_final_expected_workload",
                "minimum_num_stages",
                "maximum_retention_factor",
            ],
        },
        "mechanism_semantics": {
            "schedule_is_predeclared": True,
            "realized_compute_is_recycled": False,
            "intermediate_stages": (
                "none"
                if is_papernot
                else "conditioned_poisson_ordered_top_m_release"
            ),
            "final_stage": "unconditioned_poisson_top_1_release",
            "final_stage_k_zero": "data_independent_fallback",
            "resources_are_cumulative": True,
            "accounting_uses_incremental_resources": True,
            "common_noise_multiplier_across_stages": True,
        },
        "constraints": asdict(constraints),
        "privacy": {
            "target_epsilon": calibration.target_epsilon,
            "achieved_epsilon": calibration.achieved_epsilon,
            "delta": calibration.delta,
            "common_noise_multiplier_sigma": calibration.noise_multiplier,
            "sampling_rate": sampling_rate,
            "best_renyi_order": calibration.best_renyi_order,
            "relative_sigma_tolerance": (
                calibration.relative_sigma_tolerance
            ),
            "at_minimum_sigma": calibration.at_minimum_sigma,
            "bisection_iterations": calibration.bisection_iterations,
            "accountant_evaluations": calibration.accountant_evaluations,
            "infeasible_lower_sigma": calibration.infeasible_lower_sigma,
            "infeasible_lower_epsilon": calibration.infeasible_lower_epsilon,
            "feasible_upper_sigma": calibration.feasible_upper_sigma,
            "feasible_upper_epsilon": calibration.feasible_upper_epsilon,
        },
        "schedule": {
            "num_stages": schedule.num_stages,
            "initial_expected_workload_w_1": (
                schedule.initial_expected_num_trials
            ),
            "retention_factor_q": schedule.retention_factor,
            "resource_growth_factor_rho": schedule.resource_growth_factor,
            "minimum_resource_r_1": schedule.minimum_resource,
            "required_final_resource_r_L": schedule.required_final_resource,
            "survivor_rounding": schedule.survivor_rounding,
            "resource_rounding": schedule.resource_rounding,
            "expected_compute": schedule.expected_compute,
            "stages": [asdict(stage) for stage in schedule.stages],
        },
    }


def _validate_plot_config(plot_config: DictConfig) -> tuple[tuple[float, float], int]:
    figure_size = tuple(float(value) for value in plot_config.figsize)
    if len(figure_size) != 2 or any(value <= 0.0 for value in figure_size):
        raise ValueError("plot.figsize must contain two positive values.")
    dpi = _validate_positive_integer(plot_config.dpi, "plot.dpi")
    return (figure_size[0], figure_size[1]), dpi


def _plot_search_diagnostics(
    search_result: PrivacyCalibratedScheduleSearchResult,
    *,
    pareto_keys: set[tuple[int, float | None]],
    selected_key: tuple[int, float | None],
    output_directory: Path,
    output_config: DictConfig,
    plot_config: DictConfig,
) -> dict[str, Path]:
    # Import lazily so pure unit tests can import selection helpers without
    # initializing a Matplotlib cache.
    from matplotlib import pyplot as plt

    figure_size, dpi = _validate_plot_config(plot_config)
    feasible = search_result.feasible_candidates
    stage_counts = sorted(
        {candidate.structural_candidate.num_stages for candidate in feasible}
    )
    colors = plt.get_cmap("viridis")(
        np.linspace(0.1, 0.9, max(len(stage_counts), 1))
    )
    color_by_stage = dict(zip(stage_counts, colors))

    figure, axis = plt.subplots(figsize=figure_size)
    for num_stages in stage_counts:
        group = [
            candidate
            for candidate in feasible
            if candidate.structural_candidate.num_stages == num_stages
        ]
        axis.scatter(
            [
                candidate.structural_candidate.schedule.expected_compute
                for candidate in group
            ],
            [candidate.calibration.noise_multiplier for candidate in group],
            color=color_by_stage[num_stages],
            marker="D" if num_stages == 1 else "o",
            label=(
                "Papernot ($L=1$)"
                if num_stages == 1
                else rf"DP-SHA $L={num_stages}$"
            ),
            alpha=0.85,
        )
    for candidate in feasible:
        key = _candidate_key(candidate)
        schedule = candidate.structural_candidate.schedule
        calibration = candidate.calibration
        if schedule is None or calibration is None:
            continue
        if key in pareto_keys:
            axis.scatter(
                schedule.expected_compute,
                calibration.noise_multiplier,
                facecolors="none",
                edgecolors="black",
                linewidths=1.1,
                s=80,
            )
        if key == selected_key:
            axis.scatter(
                schedule.expected_compute,
                calibration.noise_multiplier,
                marker="*",
                color="red",
                edgecolors="black",
                linewidths=0.7,
                s=180,
                label="Selected",
                zorder=5,
            )
    axis.set_xlabel("Expected communication rounds")
    axis.set_ylabel(r"Calibrated common noise multiplier $\sigma$")
    axis.set_title("Fixed-Budget DP-SHA Schedule Search")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    sigma_compute_path = output_directory / str(
        output_config.sigma_vs_expected_compute_plot_filename
    )
    figure.savefig(sigma_compute_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=figure_size)
    multi_stage_counts = [value for value in stage_counts if value >= 2]
    for num_stages in multi_stage_counts:
        group = sorted(
            (
                candidate
                for candidate in feasible
                if candidate.structural_candidate.num_stages == num_stages
            ),
            key=lambda candidate: candidate.structural_candidate.retention_factor,
        )
        axis.plot(
            [
                candidate.structural_candidate.retention_factor
                for candidate in group
            ],
            [candidate.calibration.noise_multiplier for candidate in group],
            marker="o",
            color=color_by_stage[num_stages],
            label=rf"$L={num_stages}$",
        )
    papernot_candidates = [
        candidate
        for candidate in feasible
        if candidate.structural_candidate.num_stages == 1
    ]
    if papernot_candidates:
        papernot = papernot_candidates[0]
        papernot_label = "Papernot ($L=1$)"
        if _candidate_key(papernot) == selected_key:
            papernot_label += " — selected"
        axis.axhline(
            papernot.calibration.noise_multiplier,
            color=color_by_stage[1],
            linestyle="--",
            linewidth=1.5,
            label=papernot_label,
        )
    selected = next(
        candidate for candidate in feasible if _candidate_key(candidate) == selected_key
    )
    if selected.structural_candidate.retention_factor is not None:
        axis.scatter(
            selected.structural_candidate.retention_factor,
            selected.calibration.noise_multiplier,
            marker="*",
            color="red",
            edgecolors="black",
            linewidths=0.7,
            s=180,
            label="Selected",
            zorder=5,
        )
    axis.set_xlabel(r"Retention factor $q$")
    axis.set_ylabel(r"Calibrated common noise multiplier $\sigma$")
    axis.set_title("Privacy Cost Across Geometric Schedules")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    sigma_q_path = output_directory / str(
        output_config.sigma_vs_retention_factor_plot_filename
    )
    figure.savefig(sigma_q_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return {
        "sigma_vs_expected_compute": sigma_compute_path,
        "sigma_vs_retention_factor": sigma_q_path,
    }


def optimize_fixed_expected_budget_dp_sha(config: DictConfig) -> dict:
    """Run and persist the deterministic fixed-budget schedule search."""
    runner_start_time = time.perf_counter()
    experiment = config.experiment
    runner_name = str(experiment.runner).strip()
    if runner_name != _RUNNER_NAME:
        raise ValueError(
            f"Expected experiment.runner={_RUNNER_NAME!r}, received "
            f"{runner_name!r}."
        )

    num_stages_values = _stage_count_values(experiment.schedule)
    retention_factors = (
        _retention_factors(experiment.schedule)
        if any(value >= 2 for value in num_stages_values)
        else ()
    )
    orders = _renyi_orders(experiment.privacy)
    minimum_resource = _validate_positive_integer(
        experiment.schedule.minimum_resource,
        "schedule.minimum_resource",
    )
    required_final_resource = _validate_positive_integer(
        experiment.schedule.required_final_resource,
        "schedule.required_final_resource",
    )
    if (
        any(value >= 2 for value in num_stages_values)
        and required_final_resource <= minimum_resource
    ):
        raise ValueError(
            "schedule.required_final_resource must exceed minimum_resource."
        )
    num_configurations = _validate_positive_integer(
        experiment.coverage.num_configurations,
        "coverage.num_configurations",
    )
    coverage_criterion = str(experiment.coverage.criterion).strip().lower()
    if coverage_criterion not in {
        "all_configurations",
        "at_least_one_good",
    }:
        raise ValueError(
            "coverage.criterion must be 'all_configurations' or "
            "'at_least_one_good'."
        )
    configured_good_count = experiment.coverage.get(
        "num_good_configurations",
        None,
    )
    if coverage_criterion == "all_configurations":
        if configured_good_count is not None:
            raise ValueError(
                "coverage.num_good_configurations must be null or omitted when "
                "coverage.criterion=all_configurations."
            )
        num_good_configurations = None
    else:
        if configured_good_count is None:
            raise ValueError(
                "coverage.num_good_configurations is required when "
                "coverage.criterion=at_least_one_good."
            )
        num_good_configurations = _validate_positive_integer(
            configured_good_count,
            "coverage.num_good_configurations",
        )
        if num_good_configurations > num_configurations:
            raise ValueError(
                "coverage.num_good_configurations cannot exceed "
                "num_configurations."
            )
    target_initial_coverage = _validate_open_probability(
        experiment.coverage.target_probability,
        "coverage.target_probability",
    )
    target_final_nonempty_probability = _validate_open_probability(
        experiment.coverage.target_final_nonempty_probability,
        "coverage.target_final_nonempty_probability",
    )
    maximum_initial_expected_num_trials = _validate_positive_float(
        experiment.coverage.maximum_initial_expected_num_trials,
        "coverage.maximum_initial_expected_num_trials",
    )
    expected_compute_budget = _validate_positive_float(
        experiment.compute.expected_budget,
        "compute.expected_budget",
    )
    relative_constraint_tolerance = _validate_positive_float(
        experiment.search.relative_constraint_tolerance,
        "search.relative_constraint_tolerance",
    )
    if relative_constraint_tolerance >= 1.0:
        raise ValueError("search.relative_constraint_tolerance must be below 1.")

    privacy = experiment.privacy
    sigma_search = privacy.sigma_search
    sampling_rate = _validate_positive_float(
        config.run_settings.sampling_rate,
        "run_settings.sampling_rate",
    )
    if sampling_rate > 1.0:
        raise ValueError("run_settings.sampling_rate cannot exceed 1.")
    target_epsilon = _validate_positive_float(
        privacy.target_epsilon,
        "privacy.target_epsilon",
    )
    delta = _validate_open_probability(privacy.delta, "privacy.delta")
    initial_sigma = _validate_positive_float(
        sigma_search.initial_sigma,
        "privacy.sigma_search.initial_sigma",
    )
    minimum_sigma = _validate_positive_float(
        sigma_search.minimum_sigma,
        "privacy.sigma_search.minimum_sigma",
    )
    maximum_sigma = _validate_positive_float(
        sigma_search.maximum_sigma,
        "privacy.sigma_search.maximum_sigma",
    )
    if not minimum_sigma <= initial_sigma <= maximum_sigma:
        raise ValueError(
            "privacy.sigma_search.initial_sigma must lie between the "
            "configured minimum and maximum."
        )
    if minimum_sigma >= maximum_sigma:
        raise ValueError(
            "privacy.sigma_search.minimum_sigma must be below maximum_sigma."
        )
    relative_sigma_tolerance = _validate_positive_float(
        sigma_search.relative_tolerance,
        "privacy.sigma_search.relative_tolerance",
    )
    if relative_sigma_tolerance >= 1.0:
        raise ValueError(
            "privacy.sigma_search.relative_tolerance must be below 1."
        )

    structural_start_time = time.perf_counter()
    structural_search = enumerate_geometric_schedule_candidates(
        num_stages_values=num_stages_values,
        retention_factors=retention_factors,
        minimum_resource=minimum_resource,
        required_final_resource=required_final_resource,
        num_configurations=num_configurations,
        num_good_configurations=num_good_configurations,
        target_initial_good_coverage=target_initial_coverage,
        target_final_nonempty_probability=target_final_nonempty_probability,
        expected_compute_budget=expected_compute_budget,
        coverage_criterion=coverage_criterion,
        maximum_initial_expected_num_trials=(
            maximum_initial_expected_num_trials
        ),
        survivor_rounding=str(experiment.schedule.survivor_rounding),
        resource_rounding=str(experiment.schedule.resource_rounding),
        relative_tolerance=relative_constraint_tolerance,
    )
    structural_search_wall_time_seconds = (
        time.perf_counter() - structural_start_time
    )
    configured_num_workers = experiment.search.get("num_workers", "auto")
    multiprocessing_start_method = str(
        experiment.search.get("multiprocessing_start_method", "spawn")
    )
    requested_workers, resolved_workers, worker_source = (
        _resolve_num_workers(
            configured_num_workers,
            num_feasible_candidates=len(
                structural_search.feasible_candidates
            ),
        )
    )
    print(
        "Candidate calibration parallelism: "
        f"requested={requested_workers}, resolved={resolved_workers}, "
        f"source={worker_source}, "
        f"start_method={multiprocessing_start_method}.",
        flush=True,
    )
    calibrated_execution = _calibrate_schedule_candidates_with_workers(
        structural_search,
        sampling_rate=sampling_rate,
        target_epsilon=target_epsilon,
        delta=delta,
        orders=orders,
        initial_sigma=initial_sigma,
        minimum_sigma=minimum_sigma,
        maximum_sigma=maximum_sigma,
        relative_sigma_tolerance=relative_sigma_tolerance,
        max_iterations=_validate_positive_integer(
            sigma_search.max_iterations,
            "privacy.sigma_search.max_iterations",
        ),
        reject_renyi_order_boundary=bool(
            privacy.reject_renyi_order_boundary
        ),
        configured_num_workers=configured_num_workers,
        multiprocessing_start_method=multiprocessing_start_method,
    )
    calibrated_search = calibrated_execution.search_result

    selected = (
        select_minimum_sigma_candidate(calibrated_search)
        if calibrated_search.feasible_candidates
        else None
    )
    frontier = pareto_optimal_candidates(calibrated_search)
    pareto_keys = {_candidate_key(candidate) for candidate in frontier}
    selected_key = None if selected is None else _candidate_key(selected)
    candidate_rows = _candidate_rows(
        calibrated_search,
        pareto_keys=pareto_keys,
        selected_key=selected_key,
        execution_diagnostics=calibrated_execution.diagnostics,
    )
    stage_rows = _stage_rows(
        calibrated_search,
        selected_key=selected_key,
    )

    output_directory = Path(HydraConfig.get().runtime.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output = experiment.output
    all_candidates_path = output_directory / str(output.all_candidates_filename)
    feasible_path = output_directory / str(output.feasible_candidates_filename)
    rejected_path = output_directory / str(output.rejected_candidates_filename)
    pareto_path = output_directory / str(output.pareto_candidates_filename)
    stages_path = output_directory / str(output.stage_diagnostics_filename)
    metadata_path = output_directory / str(output.search_metadata_filename)
    selected_path = output_directory / str(output.selected_schedule_filename)

    _write_csv(all_candidates_path, candidate_rows, _CANDIDATE_FIELDS)
    _write_csv(
        feasible_path,
        (row for row in candidate_rows if row["privacy_feasible"]),
        _CANDIDATE_FIELDS,
    )
    _write_csv(
        rejected_path,
        (row for row in candidate_rows if not row["privacy_feasible"]),
        _CANDIDATE_FIELDS,
    )
    _write_csv(
        pareto_path,
        (row for row in candidate_rows if row["is_pareto_optimal"]),
        _CANDIDATE_FIELDS,
    )
    _write_csv(stages_path, stage_rows, _STAGE_FIELDS)

    metadata = {
        "schema_version": _SCHEMA_VERSION,
        "runner": _RUNNER_NAME,
        "deterministic_search": True,
        "random_seed": None,
        "num_candidates": len(calibrated_search.candidates),
        "num_structurally_feasible": sum(
            candidate.structural_candidate.is_feasible
            for candidate in calibrated_search.candidates
        ),
        "num_privacy_feasible": len(calibrated_search.feasible_candidates),
        "num_pareto_optimal": len(frontier),
        "selected_candidate_key": selected_key,
        "execution": {
            "structural_search_wall_time_seconds": (
                structural_search_wall_time_seconds
            ),
            "calibration_wall_time_seconds": (
                calibrated_execution.calibration_wall_time_seconds
            ),
            "summed_candidate_wall_time_seconds": (
                calibrated_execution.summed_candidate_wall_time_seconds
            ),
            "observed_candidate_concurrency_ratio": (
                calibrated_execution.summed_candidate_wall_time_seconds
                / calibrated_execution.calibration_wall_time_seconds
                if calibrated_execution.calibration_wall_time_seconds > 0.0
                else 0.0
            ),
            "requested_num_workers": (
                calibrated_execution.requested_num_workers
            ),
            "resolved_num_workers": (
                calibrated_execution.resolved_num_workers
            ),
            "worker_resolution_source": (
                calibrated_execution.worker_resolution_source
            ),
            "multiprocessing_start_method": (
                calibrated_execution.multiprocessing_start_method
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get(
                "SLURM_CPUS_PER_TASK"
            ),
            "parent_pid": os.getpid(),
            "wall_time_before_metadata_write_seconds": (
                time.perf_counter() - runner_start_time
            ),
        },
        "renyi_orders": {
            "minimum": orders[0],
            "maximum": orders[-1],
            "count": len(orders),
        },
        "resolved_config": OmegaConf.to_container(config, resolve=True),
    }
    _write_json(metadata_path, metadata)

    plot_paths = {}
    if selected is not None and selected_key is not None:
        selected_index = next(
            index
            for index, candidate in enumerate(
                calibrated_search.candidates,
                start=1,
            )
            if candidate is selected
        )
        selected_document = _selected_schedule_document(
            selected,
            candidate_id=_candidate_id(selected, selected_index),
            sampling_rate=sampling_rate,
        )
        _write_json(selected_path, selected_document)
        plot_paths = _plot_search_diagnostics(
            calibrated_search,
            pareto_keys=pareto_keys,
            selected_key=selected_key,
            output_directory=output_directory,
            output_config=output,
            plot_config=experiment.plot,
        )

    metadata["execution"]["total_runner_wall_time_seconds"] = (
        time.perf_counter() - runner_start_time
    )
    _write_json(metadata_path, metadata)

    print(
        "Fixed-budget DP-SHA search: "
        f"{len(calibrated_search.candidates)} candidates, "
        f"{len(calibrated_search.feasible_candidates)} privacy-feasible, "
        f"{len(frontier)} Pareto-optimal.",
        flush=True,
    )
    print(f"Saved candidate diagnostics: {all_candidates_path}", flush=True)
    print(f"Saved per-stage diagnostics: {stages_path}", flush=True)
    if selected is None:
        raise RuntimeError(
            "No privacy-feasible schedule was found. Rejection diagnostics "
            f"were saved to {rejected_path}."
        )
    print(f"Saved selected schedule: {selected_path}", flush=True)
    for plot_path in plot_paths.values():
        print(f"Saved schedule-search plot: {plot_path}", flush=True)
    return {
        "all_candidates_path": all_candidates_path,
        "feasible_candidates_path": feasible_path,
        "rejected_candidates_path": rejected_path,
        "pareto_candidates_path": pareto_path,
        "stage_diagnostics_path": stages_path,
        "metadata_path": metadata_path,
        "selected_schedule_path": selected_path,
        "plot_paths": plot_paths,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config_cl")
def main(config: DictConfig) -> None:
    optimize_fixed_expected_budget_dp_sha(config)


if __name__ == "__main__":
    main()
