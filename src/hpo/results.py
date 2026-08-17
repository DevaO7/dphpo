"""Result metadata shared by federated and centralized HPO experiments."""
import csv
import json
import math
from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt

from .planning import (
    PLAN_FILENAMES,
    generate_stage_2_plan,
    get_privacy_matched_plan_directory,
    get_privacy_matched_simulations_directory,
    load_stage_2_plan,
    load_privacy_matched_simulation_plan,
    load_simulation_plan,
    map_stage_1_plan_runs,
)

from utils.hpo_config import get_two_stage_settings


RESULT_FILENAMES = {
    "papernot_baseline": "papernot_baseline_results.JSON",
    "two_stage_tuning": "two_stage_tuning_results.JSON",
}


def get_compilation_paths(config):
    """Return the standard result directories for one experiment run."""
    run_root = (
        Path(config.output.results_root)
        / str(config.name)
        / str(config.run_id)
    )
    return {
        "run_root": run_root,
        "plan_root": run_root / "plan",
        "simulations_root": run_root / "simulations",
        "compiled_root": run_root / "compiled",
    }


def build_privacy_compute_points(config, stage_compute_schedule):
    """Build compute-matched HPO coordinates from experiment settings.

    These coordinates describe expected compute and privacy before any
    plans are sampled or simulations are run. Stage 1 uses each configured
    value in ``base_E_K_list``. Stage 2 uses its separately configured
    expected trial count, and the Papernot baseline is assigned the same
    expected compute as the corresponding two-stage point.
    """
    stage_compute_schedule = [
        float(stage_compute)
        for stage_compute in stage_compute_schedule
    ]
    if (
        len(stage_compute_schedule) != 2
        or any(
            not math.isfinite(stage_compute)
            or stage_compute <= 0.0
            for stage_compute in stage_compute_schedule
        )
    ):
        raise ValueError(
            "stage_compute_schedule must contain two finite, positive "
            "stage costs."
        )

    try:
        stage_1_expected_trials_values = [
            float(value)
            for value in config.base_E_K_list
        ]
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "base_E_K_list must contain expected Stage-1 trial counts."
        ) from error
    if not stage_1_expected_trials_values:
        raise ValueError("base_E_K_list must not be empty.")

    two_stage_settings = get_two_stage_settings(config)
    if any(
        not math.isfinite(value)
        or value <= two_stage_settings.num_survivors
        for value in stage_1_expected_trials_values
    ):
        raise ValueError(
            "Every base_E_K_list value must be finite and greater than "
            "two_stage.num_survivors because Stage 1 releases that many "
            "outputs."
        )

    stage_2_expected_trials = (
        two_stage_settings.stage_2_expected_trials
    )
    full_run_compute = sum(stage_compute_schedule)
    papernot_points = []
    two_stage_points = []

    for point_index, stage_1_expected_trials in enumerate(
        stage_1_expected_trials_values
    ):
        expected_compute = (
            stage_1_expected_trials * stage_compute_schedule[0]
            + stage_2_expected_trials * stage_compute_schedule[1]
        )
        papernot_expected_trials = expected_compute / full_run_compute
        if papernot_expected_trials < 1.0:
            raise ValueError(
                "Compute matching produced a Papernot expected trial "
                f"count below 1 at point {point_index}."
            )

        papernot_points.append(
            {
                "point_index": point_index,
                "expected_compute": expected_compute,
                "expected_num_trials": papernot_expected_trials,
            }
        )
        two_stage_points.append(
            {
                "point_index": point_index,
                "expected_compute": expected_compute,
                "stage_1_expected_num_trials": (
                    stage_1_expected_trials
                ),
                "stage_2_expected_num_trials": (
                    stage_2_expected_trials
                ),
            }
        )

    return {
        "papernot_points": papernot_points,
        "two_stage_points": two_stage_points,
        "papernot_eta": float(config.eta),
        "two_stage_eta": float(config.eta),
        "two_stage_top_m": two_stage_settings.num_survivors,
    }


def validate_privacy_order_search(dp_result, method, expected_compute):
    """Reject privacy results whose optimal order hits the search edge."""
    if dp_result.is_at_min_order or dp_result.is_at_max_order:
        boundary = (
            "minimum"
            if dp_result.is_at_min_order
            else "maximum"
        )
        raise RuntimeError(
            f"The optimal Rényi order for {method} at expected "
            f"compute {expected_compute} is the {boundary} stored "
            f"order ({dp_result.best_order}). Expand the configured "
            "Rényi-order range before reporting epsilon."
        )


def save_privacy_compute_rows(rows, csv_path):
    """Save validated privacy-versus-compute rows to CSV."""
    if not rows:
        raise ValueError("No privacy-compute result rows were produced.")
    with Path(csv_path).open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

def load_compilation_plan(config, method):
    if method not in RESULT_FILENAMES:
        raise ValueError(
            f"Unknown compilation method {method!r}."
        )
    plan_filename = PLAN_FILENAMES[method][2]
    plan = load_stage_2_plan(
        config,
        plan_filename=plan_filename,
    )
    if plan.get("method") != method:
        raise ValueError(
            f"Plan {plan_filename} has method "
            f"{plan.get('method')!r}, expected {method!r}."
        )
    metadata_checks = {
        "run_id": str(config.run_id),
        "plan_seed": int(config.seed),
        "hp_configuration_ids": [
            str(hp_id) for hp_id in config.hp_configuration_ids
        ],
        "eta": float(config.eta),
    }
    for key, expected_value in metadata_checks.items():
        if plan.get(key) != expected_value:
            raise ValueError(
                f"Compilation plan {plan_filename} has {key}="
                f"{plan.get(key)!r}, but the experiment requires "
                f"{expected_value!r}. Regenerate the plan or restore "
                "the matching experiment configuration."
            )
    points = plan.get("points")
    if (
        not isinstance(points, list)
        or len(points) != len(config.base_E_K_list)
    ):
        raise ValueError(
            f"Compilation plan {plan_filename} does not match the "
            "number of configured base_E_K_list points."
        )
    expected_num_trials = int(config.num_trials)
    for point_index, point in enumerate(points):
        trials = point.get("trials") if isinstance(point, dict) else None
        if (
            not isinstance(trials, list)
            or len(trials) != expected_num_trials
        ):
            raise ValueError(
                f"Compilation plan {plan_filename} point "
                f"{point_index} does not contain the configured "
                f"{expected_num_trials} trials."
            )
    return plan


def validate_compilation_metric_files(config, plans):
    """Fail once with all missing metric artifacts for both methods."""
    simulations_root = get_compilation_paths(config)["simulations_root"]
    missing_paths = set()
    for method, plan in plans.items():
        execution_summary = plan.get("execution_summary")
        if not isinstance(execution_summary, dict):
            raise ValueError(
                f"The {method} compilation plan has no execution summary."
            )
        specs = execution_summary.get("required_stage_2_run_specs")
        if not isinstance(specs, list) or not specs:
            raise ValueError(
                f"The {method} compilation plan has no required "
                "Stage-2 run specifications."
            )
        for spec_index, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise ValueError(
                    f"The {method} Stage-2 run specification "
                    f"{spec_index} is invalid."
                )
            try:
                stage_1_directory = str(
                    spec["stage_1_run_directory"]
                )
                stage_2_directory = str(
                    spec["stage_2_run_directory"]
                )
            except KeyError as error:
                raise ValueError(
                    f"The {method} Stage-2 run specification "
                    f"{spec_index} is missing {error.args[0]!r}."
                ) from error
            stage_1_path = (
                simulations_root / stage_1_directory / "stage_1.csv"
            )
            stage_2_path = (
                simulations_root / stage_2_directory / "stage_2.csv"
            )
            if not stage_1_path.is_file():
                missing_paths.add(stage_1_path)
            if not stage_2_path.is_file():
                missing_paths.add(stage_2_path)

    if missing_paths:
        ordered_paths = sorted(str(path) for path in missing_paths)
        display_limit = 20
        displayed_paths = ordered_paths[:display_limit]
        remainder = len(ordered_paths) - len(displayed_paths)
        details = "\n".join(
            f"  - {path}" for path in displayed_paths
        )
        if remainder:
            details += f"\n  - ... and {remainder} more"
        raise FileNotFoundError(
            "Cannot compile both HPO methods because "
            f"{len(ordered_paths)} required metric files are missing:\n"
            f"{details}"
        )




def get_metric_selection_mode(
    metric,
    evaluation_mode,
):
    normalized_mode = str(evaluation_mode).strip().lower()
    if normalized_mode in {"min", "max"}:
        return normalized_mode

    if normalized_mode != "last_round":
        raise ValueError(
            "evaluation.mode must be 'min', 'max', or "
            f"'last_round'; got {evaluation_mode!r}."
        )

    normalized_metric = (
        str(metric).strip().lower().replace(" ", "_")
    )
    if normalized_metric.endswith("_loss"):
        return "min"
    if normalized_metric.endswith(("_accuracy", "_acc")):
        return "max"

    raise ValueError(
        "Cannot infer whether a last-round metric should be "
        f"minimized or maximized from {metric!r}."
    )


def normalize_metric_name(metric):
    normalized_metric = (
        str(metric).strip().lower().replace(" ", "_")
    )
    if not normalized_metric:
        raise ValueError("Metric names must not be empty.")
    return normalized_metric

def get_evaluation_settings(config):
    evaluation = config.evaluation
    selection_config = evaluation.get("selection")
    if selection_config is None:
        selection_metric = evaluation.get("metric")
        evaluation_mode = evaluation.get("mode")
    else:
        selection_metric = selection_config.get("metric")
        evaluation_mode = selection_config.get("mode")

    if selection_metric is None or evaluation_mode is None:
        raise ValueError(
            "evaluation.selection must define both 'metric' and "
            "'mode'."
        )

    selection_metric = normalize_metric_name(
        selection_metric
    )
    evaluation_mode = str(evaluation_mode).strip().lower()
    selection_mode = get_metric_selection_mode(
        metric=selection_metric,
        evaluation_mode=evaluation_mode,
    )

    utility_config = evaluation.get("utility")
    utility_at = "selection_round"
    if utility_config is None:
        utility_metrics = [selection_metric]
    elif isinstance(utility_config, str):
        utility_metrics = [utility_config]
    else:
        utility_metrics = utility_config.get(
            "metrics",
            [selection_metric],
        )
        utility_at = str(
            utility_config.get("at", "selection_round")
        )

    if isinstance(utility_metrics, str):
        utility_metrics = [utility_metrics]
    utility_metrics = list(
        dict.fromkeys(
            normalize_metric_name(metric)
            for metric in utility_metrics
        )
    )
    if not utility_metrics:
        raise ValueError(
            "evaluation.utility.metrics must not be empty."
        )
    if utility_at != "selection_round":
        raise ValueError(
            "evaluation.utility.at currently supports only "
            f"'selection_round'; got {utility_at!r}."
        )

    return {
        "selection_metric": selection_metric,
        "evaluation_mode": evaluation_mode,
        "selection_mode": selection_mode,
        "utility_metrics": utility_metrics,
        "utility_at": utility_at,
    }

def get_compiled_evaluation_metadata(
    evaluation,
    stage_1_end,
    stage_2_end,
    evaluation_interval=1,
):
    return {
        "selection": {
            "metric": evaluation["selection_metric"],
            "mode": evaluation["evaluation_mode"],
            "selection_mode": evaluation["selection_mode"],
        },
        "utility": {
            "metrics": list(evaluation["utility_metrics"]),
            "at": evaluation["utility_at"],
        },
        "trajectory": {
            "evaluation_interval": int(evaluation_interval),
            "stage_1_rounds": [0, stage_1_end - 1],
            "stage_2_rounds": [
                stage_1_end,
                stage_2_end - 1,
            ],
        },
    }

def load_metric_segment(
    csv_path,
    stage,
    expected_start_round,
    expected_final_round,
    metrics,
    evaluation_interval=1,
):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Stage-{stage} metrics CSV does not exist: {csv_path}"
        )
    if expected_final_round < expected_start_round:
        raise ValueError(
            "Expected metric round range must be non-empty; got "
            f"{expected_start_round} through {expected_final_round}."
        )
    if (
        isinstance(evaluation_interval, bool)
        or not isinstance(evaluation_interval, (int, np.integer))
        or int(evaluation_interval) <= 0
    ):
        raise ValueError(
            "evaluation_interval must be a positive integer; got "
            f"{evaluation_interval!r}."
        )
    evaluation_interval = int(evaluation_interval)

    normalized_metrics = list(
        dict.fromkeys(
            normalize_metric_name(metric)
            for metric in metrics
        )
    )
    try:
        with csv_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError(
                    f"Metrics CSV has no header: {csv_path}"
                )

            normalized_columns = {
                normalize_metric_name(column): column
                for column in fieldnames
            }
            required_columns = ["round", *normalized_metrics]
            missing_columns = [
                column
                for column in required_columns
                if column not in normalized_columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Metrics {missing_columns!r} are not available "
                    f"in {csv_path}. Available columns: "
                    f"{', '.join(fieldnames)}."
                )

            rows = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    round_number = int(
                        row[normalized_columns["round"]]
                    )
                    metric_values = {
                        metric: float(
                            row[normalized_columns[metric]]
                        )
                        for metric in normalized_metrics
                    }
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Metrics CSV contains an invalid round or "
                        f"metric at row {row_number}: {csv_path}."
                    ) from error

                invalid_values = {
                    metric: value
                    for metric, value in metric_values.items()
                    if not np.isfinite(value)
                }
                if invalid_values:
                    raise ValueError(
                        "Metrics must be finite at row "
                        f"{row_number} in {csv_path}; got "
                        f"{invalid_values!r}."
                    )

                rows.append(
                    {
                        "stage": int(stage),
                        "round": round_number,
                        "metrics": metric_values,
                    }
                )
    except OSError as error:
        raise OSError(
            f"Could not read metrics CSV: {csv_path}"
        ) from error

    expected_rounds = [
        round_number
        for round_number in range(
            expected_start_round,
            expected_final_round + 1,
        )
        if (round_number + 1) % evaluation_interval == 0
    ]
    if expected_final_round not in expected_rounds:
        expected_rounds.append(expected_final_round)
    observed_rounds = [row["round"] for row in rows]
    if observed_rounds != expected_rounds:
        raise ValueError(
            f"Stage-{stage} metrics CSV must contain exactly the "
            "configured evaluation rounds from "
            f"{expected_start_round} through {expected_final_round} "
            f"at interval {evaluation_interval}: {csv_path}. "
            f"Expected rounds: {expected_rounds!r}. Observed rounds: "
            f"{observed_rounds!r}."
        )

    return rows



def evaluate_metric_trajectory(
    metric_segments,
    selection_metric,
    evaluation_mode,
    utility_metrics,
):
    selection_metric = normalize_metric_name(selection_metric)
    utility_metrics = list(
        dict.fromkeys(
            normalize_metric_name(metric)
            for metric in utility_metrics
        )
    )
    requested_metrics = list(
        dict.fromkeys([selection_metric, *utility_metrics])
    )

    trajectory_rows = []
    metrics_paths = {}
    for segment in metric_segments:
        stage = int(segment["stage"])
        if stage in metrics_paths:
            raise ValueError(
                f"Metric trajectory contains Stage {stage} twice."
            )
        metrics_path = Path(segment["csv_path"])
        metrics_paths[stage] = str(metrics_path)
        trajectory_rows.extend(
            load_metric_segment(
                csv_path=metrics_path,
                stage=stage,
                expected_start_round=int(
                    segment["expected_start_round"]
                ),
                expected_final_round=int(
                    segment["expected_final_round"]
                ),
                metrics=requested_metrics,
                evaluation_interval=int(
                    segment.get("evaluation_interval", 1)
                ),
            )
        )

    if not trajectory_rows:
        raise ValueError(
            "Metric trajectory must contain at least one segment."
        )

    observed_rounds = [row["round"] for row in trajectory_rows]
    if observed_rounds != sorted(observed_rounds):
        raise ValueError(
            "Metric trajectory segments must be ordered by round."
        )
    if len(observed_rounds) != len(set(observed_rounds)):
        raise ValueError(
            "Metric trajectory segments contain overlapping rounds."
        )

    normalized_mode = str(evaluation_mode).strip().lower()
    if normalized_mode == "min":
        selected_row = min(
            trajectory_rows,
            key=lambda row: row["metrics"][selection_metric],
        )
    elif normalized_mode == "max":
        selected_row = max(
            trajectory_rows,
            key=lambda row: row["metrics"][selection_metric],
        )
    elif normalized_mode == "last_round":
        selected_row = trajectory_rows[-1]
    else:
        raise ValueError(
            "evaluation mode must be 'min', 'max', or "
            f"'last_round'; got {evaluation_mode!r}."
        )

    return {
        "selection": {
            "metric": selection_metric,
            "mode": normalized_mode,
            "selection_mode": get_metric_selection_mode(
                metric=selection_metric,
                evaluation_mode=normalized_mode,
            ),
            "stage": selected_row["stage"],
            "round": selected_row["round"],
            "score": selected_row["metrics"][selection_metric],
        },
        "utility": {
            metric: selected_row["metrics"][metric]
            for metric in utility_metrics
        },
        "metrics_paths": metrics_paths,
    }

def score_complete_run(
    run,
    simulations_root,
    evaluation,
    stage_1_end,
    stage_2_end,
    score_cache,
    evaluation_interval=1,
):
    stage_1_directory = str(
        run["stage_1_run_directory"]
    )
    stage_2_directory = str(
        run["stage_2_run_directory"]
    )
    cache_key = (
        stage_1_directory,
        stage_2_directory,
        evaluation["selection_metric"],
        evaluation["evaluation_mode"],
        tuple(evaluation["utility_metrics"]),
        int(evaluation_interval),
    )

    if cache_key not in score_cache:
        stage_1_metrics_path = (
            simulations_root
            / stage_1_directory
            / "stage_1.csv"
        )
        stage_2_metrics_path = (
            simulations_root
            / stage_2_directory
            / "stage_2.csv"
        )
        trajectory_evaluation = evaluate_metric_trajectory(
            metric_segments=[
                {
                    "stage": 1,
                    "csv_path": stage_1_metrics_path,
                    "expected_start_round": 0,
                    "expected_final_round": stage_1_end - 1,
                    "evaluation_interval": evaluation_interval,
                },
                {
                    "stage": 2,
                    "csv_path": stage_2_metrics_path,
                    "expected_start_round": stage_1_end,
                    "expected_final_round": stage_2_end - 1,
                    "evaluation_interval": evaluation_interval,
                },
            ],
            selection_metric=evaluation["selection_metric"],
            evaluation_mode=evaluation["evaluation_mode"],
            utility_metrics=evaluation["utility_metrics"],
        )
        score_cache[cache_key] = {
            "selection": trajectory_evaluation["selection"],
            "utility": trajectory_evaluation["utility"],
            "stage_1_metrics_path": str(stage_1_metrics_path),
            "stage_2_metrics_path": str(stage_2_metrics_path),
        }

    scored_run = copy.deepcopy(run)
    scored_run.update(
        copy.deepcopy(score_cache[cache_key])
    )
    return scored_run

def select_top_m_runs(
    sampled_runs,
    m,
    mode,
    tie_break_seed,
    score_getter,
):
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)):
        raise ValueError(
            f"top-m must be an integer; got {m!r}."
        )
    m = int(m)
    if m <= 0:
        raise ValueError(
            f"top-m must be positive; got {m!r}."
        )
    if len(sampled_runs) < m:
        raise ValueError(
            f"Cannot select top-{m} from only "
            f"{len(sampled_runs)} runs."
        )

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"min", "max"}:
        raise ValueError(
            "Selection mode must be either 'min' or 'max'; "
            f"got {mode!r}."
        )

    scores = []
    for run_index, sampled_run in enumerate(sampled_runs):
        score = float(score_getter(sampled_run))
        if not np.isfinite(score):
            raise ValueError(
                f"Run {run_index} has a non-finite selection "
                f"score: {score!r}."
            )
        scores.append(score)

    normalized_seed = [int(component) for component in tie_break_seed]
    rng = np.random.default_rng(
        np.random.SeedSequence(normalized_seed)
    )
    tie_break_values = rng.random(len(sampled_runs))

    def score_key(index):
        primary_score = scores[index]
        if normalized_mode == "max":
            primary_score = -primary_score
        return primary_score, tie_break_values[index]

    ranked_indices = sorted(
        range(len(sampled_runs)),
        key=score_key,
    )
    selected_runs = []
    for rank, sampled_index in enumerate(
        ranked_indices[:m],
        start=1,
    ):
        selected_run = copy.deepcopy(
            sampled_runs[sampled_index]
        )
        selected_run["selection_rank"] = rank
        selected_runs.append(selected_run)

    return selected_runs

def select_top_m_stage_1_runs(
    stage_1_plan,
    m,
    mode,
    seed,
):
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)):
        raise ValueError(
            f"top-m must be an integer; got {m!r}."
        )
    m = int(m)
    if m <= 0:
        raise ValueError(
            f"top-m must be positive; got {m!r}."
        )
    stage_1_plan["stage_1_top_m"] = m

    normalized_mode = str(mode).strip().lower()

    for point_index, point in enumerate(
        stage_1_plan["points"]
    ):
        for trial_index, trial in enumerate(point["trials"]):
            sampled_runs = trial["sampled_stage_1_runs"]

            trial_id = int(trial.get("trial", trial_index))
            tie_break_seed = [
                int(seed),
                point_index,
                trial_id,
            ]
            trial["selection_tie_break_seed"] = tie_break_seed
            try:
                trial["top_m_stage_1_runs"] = select_top_m_runs(
                    sampled_runs=sampled_runs,
                    m=m,
                    mode=normalized_mode,
                    tie_break_seed=tie_break_seed,
                    score_getter=lambda run: run[
                        "evaluation_score"
                    ],
                )
            except ValueError as error:
                raise ValueError(
                    "Could not select Stage-1 survivors at point "
                    f"{point_index}, trial {trial_index}: {error}"
                ) from error

    return stage_1_plan

def summarize_trial_utilities(trials, utility_metrics):
    summary = {}
    for metric in utility_metrics:
        values = np.asarray(
            [
                trial["final_selected_run"]["utility"][metric]
                for trial in trials
            ],
            dtype=float,
        )
        if values.size == 0:
            raise ValueError(
                "Cannot summarize a point with no trials."
            )
        standard_deviation = (
            float(np.std(values, ddof=1))
            if values.size > 1
            else 0.0
        )
        standard_error = (
            standard_deviation / np.sqrt(values.size)
        )
        summary[metric] = {
            "num_trials": int(values.size),
            "mean": float(np.mean(values)),
            "standard_deviation": standard_deviation,
            "standard_error": float(standard_error),
            "ci95_half_width": float(
                1.96 * standard_error
            ),
        }
    return summary


def compile_papernot_results(
    config,
    plan,
    evaluation,
    stage_compute_schedule,
    evaluation_interval=1,
):
    result = copy.deepcopy(plan)
    result["result_type"] = "compiled_hpo_results"
    result["evaluation"] = get_compiled_evaluation_metadata(
        evaluation=evaluation,
        stage_1_end=int(config.simulation.stage_1_end),
        stage_2_end=int(config.simulation.stage_2_end),
        evaluation_interval=evaluation_interval,
    )
    simulations_root = get_compilation_paths(config)[
        "simulations_root"
    ]
    stage_1_end = int(config.simulation.stage_1_end)
    stage_2_end = int(config.simulation.stage_2_end)
    score_cache = {}

    for point_index, point in enumerate(result["points"]):
        point["expected_compute"] = float(
            float(point["E_K"]) * sum(stage_compute_schedule)
        )
        for trial_index, trial in enumerate(point["trials"]):
            scored_runs = [
                score_complete_run(
                    run=run,
                    simulations_root=simulations_root,
                    evaluation=evaluation,
                    stage_1_end=stage_1_end,
                    stage_2_end=stage_2_end,
                    score_cache=score_cache,
                    evaluation_interval=evaluation_interval,
                )
                for run in trial["sampled_stage_2_runs"]
            ]
            trial["sampled_stage_2_runs"] = scored_runs
            trial_id = int(trial.get("trial", trial_index))
            tie_break_seed = [
                int(config.seed),
                3,
                0,
                point_index,
                trial_id,
            ]
            selected_runs = select_top_m_runs(
                sampled_runs=scored_runs,
                m=1,
                mode=evaluation["selection_mode"],
                tie_break_seed=tie_break_seed,
                score_getter=lambda run: run["selection"][
                    "score"
                ],
            )
            trial["selection_tie_break_seed"] = tie_break_seed
            trial["final_selected_run"] = selected_runs[0]

        point["aggregate_utility"] = (
            summarize_trial_utilities(
                trials=point["trials"],
                utility_metrics=evaluation["utility_metrics"],
            )
        )

    return result

def build_stage_2_spec_lookup(plan):
    lookup = {}
    required_specs = plan["execution_summary"][
        "required_stage_2_run_specs"
    ]
    for spec_index, spec in enumerate(required_specs):
        run_directory = str(spec["stage_2_run_directory"])
        if run_directory in lookup:
            raise ValueError(
                "Stage-2 plan contains duplicate required run "
                f"directory {run_directory!r} at entry "
                f"{spec_index}."
            )
        lookup[run_directory] = dict(spec)
    return lookup

def save_trial_result_rows(rows, csv_path):
    if not rows:
        raise ValueError("No trial result rows were compiled.")
    csv_path = Path(csv_path)
    temporary_path = csv_path.with_suffix(
        f"{csv_path.suffix}.tmp"
    )
    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0]),
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(csv_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def plot_expected_compute_utility(
    rows,
    output_directory,
    compute_axis_label=(
        "Expected compute (communication rounds × local updates)"
    ),
):
    method_labels = {
        "papernot_baseline": "Papernot baseline",
        "two_stage_tuning": "Two-stage tuning",
    }
    utility_metrics = sorted(
        {row["utility_metric"] for row in rows}
    )
    plot_paths = []

    for utility_metric in utility_metrics:
        figure, axis = plt.subplots(figsize=(7, 5))
        for method in RESULT_FILENAMES:
            method_rows = [
                row
                for row in rows
                if row["method"] == method
                and row["utility_metric"] == utility_metric
            ]
            values_by_compute = {}
            for row in method_rows:
                values_by_compute.setdefault(
                    float(row["expected_compute"]),
                    [],
                ).append(float(row["utility_score"]))

            expected_compute_values = sorted(values_by_compute)
            means = []
            ci95_half_widths = []
            for expected_compute in expected_compute_values:
                values = np.asarray(
                    values_by_compute[expected_compute],
                    dtype=float,
                )
                standard_deviation = (
                    float(np.std(values, ddof=1))
                    if values.size > 1
                    else 0.0
                )
                means.append(float(np.mean(values)))
                ci95_half_widths.append(
                    1.96
                    * standard_deviation
                    / np.sqrt(values.size)
                )

            means = np.asarray(means, dtype=float)
            ci95_half_widths = np.asarray(
                ci95_half_widths,
                dtype=float,
            )
            mean_line, = axis.plot(
                expected_compute_values,
                means,
                marker="o",
                label=method_labels[method],
            )
            axis.fill_between(
                expected_compute_values,
                means - ci95_half_widths,
                means + ci95_half_widths,
                color=mean_line.get_color(),
                alpha=0.2,
                linewidth=0,
            )

        axis.set_xlabel(str(compute_axis_label))
        axis.set_ylabel(
            utility_metric.replace("_", " ").title()
        )
        axis.set_title(
            f"{utility_metric.replace('_', ' ').title()} "
            "vs Expected Compute"
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        plot_path = (
            output_directory
            / f"expected_compute_vs_{utility_metric}.png"
        )
        figure.savefig(plot_path, dpi=300)
        plt.close(figure)
        plot_paths.append(plot_path)

    return plot_paths

def build_trial_result_rows(compiled_results):
    rows = []
    for method, result in compiled_results.items():
        utility_metrics = result["evaluation"]["utility"][
            "metrics"
        ]
        for point_index, point in enumerate(result["points"]):
            for trial_index, trial in enumerate(point["trials"]):
                selected_run = trial["final_selected_run"]
                if method == "papernot_baseline":
                    stage_1_E_K = point["E_K"]
                    stage_2_E_K = point["E_K"]
                    stage_1_sampled_K = trial["sampled_K"]
                    stage_2_sampled_K = trial["sampled_K"]
                else:
                    stage_1_E_K = point["stage_1_E_K"]
                    stage_2_E_K = point["stage_2_E_K"]
                    stage_1_sampled_K = trial[
                        "trial_stage_1"
                    ]["sampled_K"]
                    stage_2_sampled_K = trial[
                        "trial_stage_2"
                    ]["sampled_K"]

                for utility_metric in utility_metrics:
                    rows.append(
                        {
                            "method": method,
                            "point_index": point_index,
                            "trial": int(
                                trial.get("trial", trial_index)
                            ),
                            "expected_compute": point[
                                "expected_compute"
                            ],
                            "stage_1_E_K": stage_1_E_K,
                            "stage_2_E_K": stage_2_E_K,
                            "stage_1_sampled_K": (
                                stage_1_sampled_K
                            ),
                            "stage_2_sampled_K": (
                                stage_2_sampled_K
                            ),
                            "hp_configuration_id": selected_run[
                                "hp_configuration_id"
                            ],
                            "stage_1_run_index": selected_run[
                                "stage_1_run_index"
                            ],
                            "continuation_index": selected_run[
                                "continuation_index"
                            ],
                            "selection_metric": selected_run[
                                "selection"
                            ]["metric"],
                            "selection_mode": selected_run[
                                "selection"
                            ]["mode"],
                            "selection_stage": selected_run[
                                "selection"
                            ]["stage"],
                            "selection_round": selected_run[
                                "selection"
                            ]["round"],
                            "selection_score": selected_run[
                                "selection"
                            ]["score"],
                            "utility_metric": utility_metric,
                            "utility_score": selected_run[
                                "utility"
                            ][utility_metric],
                            "stage_1_metrics_path": selected_run[
                                "stage_1_metrics_path"
                            ],
                            "stage_2_metrics_path": selected_run[
                                "stage_2_metrics_path"
                            ],
                        }
                    )
    return rows

def compile_two_stage_results(
    config,
    plan,
    evaluation,
    stage_compute_schedule,
    evaluation_interval=1,
):
    result = copy.deepcopy(plan)
    result["result_type"] = "compiled_hpo_results"
    result["evaluation"] = get_compiled_evaluation_metadata(
        evaluation=evaluation,
        stage_1_end=int(config.simulation.stage_1_end),
        stage_2_end=int(config.simulation.stage_2_end),
        evaluation_interval=evaluation_interval,
    )
    simulations_root = get_compilation_paths(config)[
        "simulations_root"
    ]
    stage_1_end = int(config.simulation.stage_1_end)
    stage_2_end = int(config.simulation.stage_2_end)
    stage_2_spec_lookup = build_stage_2_spec_lookup(plan)
    score_cache = {}

    for point_index, point in enumerate(result["points"]):
        point["expected_compute"] = float(
            float(point["stage_1_E_K"])
            * stage_compute_schedule[0]
            + float(point["stage_2_E_K"])
            * stage_compute_schedule[1]
        )
        for trial_index, trial in enumerate(point["trials"]):
            stage_2_trial = trial["trial_stage_2"]
            scored_runs = []
            for sampled_run in stage_2_trial[
                "sampled_stage_2_runs"
            ]:
                run_directory = str(
                    sampled_run["stage_2_run_directory"]
                )
                try:
                    run_spec = stage_2_spec_lookup[run_directory]
                except KeyError as error:
                    raise ValueError(
                        "Sampled Stage-2 run is missing from the "
                        "execution summary: "
                        f"{run_directory!r}."
                    ) from error
                complete_run = {
                    **run_spec,
                    "sample_index": int(
                        sampled_run["sample_index"]
                    ),
                }
                scored_runs.append(
                    score_complete_run(
                        run=complete_run,
                        simulations_root=simulations_root,
                        evaluation=evaluation,
                        stage_1_end=stage_1_end,
                        stage_2_end=stage_2_end,
                        score_cache=score_cache,
                        evaluation_interval=evaluation_interval,
                    )
                )

            stage_2_trial["sampled_stage_2_runs"] = scored_runs
            trial_id = int(trial.get("trial", trial_index))
            tie_break_seed = [
                int(config.seed),
                3,
                1,
                point_index,
                trial_id,
            ]
            selected_runs = select_top_m_runs(
                sampled_runs=scored_runs,
                m=1,
                mode=evaluation["selection_mode"],
                tie_break_seed=tie_break_seed,
                score_getter=lambda run: run["selection"][
                    "score"
                ],
            )
            trial["selection_tie_break_seed"] = tie_break_seed
            trial["final_selected_run"] = selected_runs[0]

        point["aggregate_utility"] = (
            summarize_trial_utilities(
                trials=point["trials"],
                utility_metrics=evaluation["utility_metrics"],
            )
        )

    return result


def _save_json_atomically(value, path):
    path = Path(path)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            encoder = json.JSONEncoder(
                indent=4,
                allow_nan=False,
            )
            pending_characters = 0
            for chunk in encoder.iterencode(value):
                file.write(chunk)
                pending_characters += len(chunk)
                if pending_characters >= 1_000_000:
                    file.flush()
                    pending_characters = 0
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def compile_experiment_results(
    config,
    *,
    stage_compute_schedule,
    evaluation_interval=1,
    compute_axis_label=(
        "Expected compute (communication rounds × local updates)"
    ),
):
    """Compile both HPO methods and persist trial-level artifacts."""
    if (
        isinstance(evaluation_interval, bool)
        or not isinstance(evaluation_interval, (int, np.integer))
        or int(evaluation_interval) <= 0
    ):
        raise ValueError(
            "evaluation_interval must be a positive integer; got "
            f"{evaluation_interval!r}."
        )
    evaluation_interval = int(evaluation_interval)
    stage_compute_schedule = np.asarray(
        stage_compute_schedule,
        dtype=float,
    )
    if (
        stage_compute_schedule.shape != (2,)
        or not np.all(np.isfinite(stage_compute_schedule))
        or np.any(stage_compute_schedule <= 0.0)
    ):
        raise ValueError(
            "stage_compute_schedule must contain two finite, positive "
            "stage costs."
        )
    stage_compute_schedule = stage_compute_schedule.tolist()

    evaluation = get_evaluation_settings(config)
    paths = get_compilation_paths(config)
    plans = {
        method: load_compilation_plan(config, method)
        for method in RESULT_FILENAMES
    }
    validate_compilation_metric_files(config, plans)
    compiled_results = {
        "papernot_baseline": compile_papernot_results(
            config=config,
            plan=plans["papernot_baseline"],
            evaluation=evaluation,
            stage_compute_schedule=stage_compute_schedule,
            evaluation_interval=evaluation_interval,
        ),
        "two_stage_tuning": compile_two_stage_results(
            config=config,
            plan=plans["two_stage_tuning"],
            evaluation=evaluation,
            stage_compute_schedule=stage_compute_schedule,
            evaluation_interval=evaluation_interval,
        ),
    }

    papernot_compute = np.asarray(
        [
            point["expected_compute"]
            for point in compiled_results["papernot_baseline"]["points"]
        ],
        dtype=float,
    )
    two_stage_compute = np.asarray(
        [
            point["expected_compute"]
            for point in compiled_results["two_stage_tuning"]["points"]
        ],
        dtype=float,
    )
    if (
        papernot_compute.shape != two_stage_compute.shape
        or not np.allclose(
            papernot_compute,
            two_stage_compute,
            rtol=1e-12,
            atol=1e-9,
        )
    ):
        raise ValueError(
            "Papernot and two-stage plans do not have matching "
            "expected-compute coordinates. Regenerate both plans."
        )

    compiled_root = paths["compiled_root"]
    compiled_root.mkdir(parents=True, exist_ok=True)
    result_paths = {}
    for method, result in compiled_results.items():
        plan_path = paths["plan_root"] / PLAN_FILENAMES[method][2]
        result["source_plan_path"] = str(plan_path)
        result_path = compiled_root / RESULT_FILENAMES[method]
        _save_json_atomically(result, result_path)
        result_paths[method] = result_path

    trial_rows = build_trial_result_rows(compiled_results)
    trial_csv_path = compiled_root / "trial_results.csv"
    save_trial_result_rows(
        rows=trial_rows,
        csv_path=trial_csv_path,
    )
    plot_paths = plot_expected_compute_utility(
        rows=trial_rows,
        output_directory=compiled_root,
        compute_axis_label=compute_axis_label,
    )

    return {
        "result_paths": result_paths,
        "trial_csv_path": trial_csv_path,
        "plot_paths": plot_paths,
    }


def load_stage_1_metric(
    csv_path,
    metric,
    evaluation_mode,
    expected_final_round,
    evaluation_interval=1,
):
    evaluation = evaluate_metric_trajectory(
        metric_segments=[
            {
                "stage": 1,
                "csv_path": csv_path,
                "expected_start_round": 0,
                "expected_final_round": expected_final_round,
                "evaluation_interval": evaluation_interval,
            }
        ],
        selection_metric=metric,
        evaluation_mode=evaluation_mode,
        utility_metrics=[metric],
    )
    return {
        "evaluation_round": evaluation["selection"]["round"],
        "evaluation_score": evaluation["selection"]["score"],
    }


def record_stage_1_run_scores(
    stage_1_plan,
    config,
    evaluation_interval=1,
    simulations_root=None,
):
    evaluation = get_evaluation_settings(config)
    metric = evaluation["selection_metric"]
    evaluation_mode = evaluation["evaluation_mode"]
    expected_final_round = (
        int(config.simulation.stage_1_end) - 1
    )
    if expected_final_round < 0:
        raise ValueError(
            "simulation.stage_1_end must be positive; "
            f"got {config.simulation.stage_1_end!r}."
        )
    if (
        isinstance(evaluation_interval, bool)
        or not isinstance(evaluation_interval, (int, np.integer))
        or int(evaluation_interval) <= 0
    ):
        raise ValueError(
            "evaluation_interval must be a positive integer; got "
            f"{evaluation_interval!r}."
        )
    evaluation_interval = int(evaluation_interval)

    stage_1_plan["evaluation"] = {
        "metric": metric,
        "mode": evaluation_mode,
        "selection_mode": get_metric_selection_mode(
            metric=metric,
            evaluation_mode=evaluation_mode,
        ),
        "expected_final_round": expected_final_round,
        "evaluation_interval": evaluation_interval,
    }

    if simulations_root is None:
        simulations_root = (
            Path(config.output.results_root)
            / str(config.name)
            / str(config.run_id)
            / "simulations"
        )
    else:
        simulations_root = Path(simulations_root)
    score_cache = {}

    for point in stage_1_plan["points"]:
        for trial in point["trials"]:
            for sampled_run in trial["sampled_stage_1_runs"]:
                run_directory = sampled_run[
                    "stage_1_run_directory"
                ]
                csv_path = (
                    simulations_root
                    / run_directory
                    / "stage_1.csv"
                )

                if run_directory not in score_cache:
                    score_cache[run_directory] = (
                        load_stage_1_metric(
                            csv_path=csv_path,
                            metric=metric,
                            evaluation_mode=evaluation_mode,
                            expected_final_round=(
                                expected_final_round
                            ),
                            evaluation_interval=(
                                evaluation_interval
                            ),
                        )
                    )

                sampled_run.update(score_cache[run_directory])
                sampled_run["stage_1_metrics_path"] = str(
                    csv_path
                )

    return stage_1_plan


def generate_stage_2_plan_from_results(
    config,
    *,
    evaluation_interval=1,
):
    """Score Stage-1 runs, select top-m, and persist the Stage-2 plan."""
    if str(config.simulation.method) != "two_stage_tuning":
        raise ValueError(
            "Stage-2 plan generation requires "
            "simulation.method='two_stage_tuning'."
        )

    two_stage_settings = get_two_stage_settings(config)
    stage_1_plan = load_simulation_plan(config, stage=1)
    stage_1_plan = map_stage_1_plan_runs(
        stage_1_plan,
        config.hp_configuration_ids,
    )
    stage_1_plan = record_stage_1_run_scores(
        stage_1_plan,
        config,
        evaluation_interval=evaluation_interval,
    )
    stage_1_plan = select_top_m_stage_1_runs(
        stage_1_plan,
        m=two_stage_settings.num_survivors,
        mode=stage_1_plan["evaluation"]["selection_mode"],
        seed=config.seed,
    )
    return generate_stage_2_plan(stage_1_plan, config)


def generate_privacy_matched_stage_2_plan_from_results(
    config,
    *,
    evaluation_interval=1,
):
    """Generate one nested Stage-2 plan from a completed Stage 1."""
    exp_config = config.experiment
    method = str(exp_config.simulation.method)
    if method != "two_stage_tuning":
        raise ValueError(
            "Privacy-matched Stage-2 plan generation requires "
            "simulation.method='two_stage_tuning'."
        )

    target_epsilon = float(
        exp_config.simulation.target_epsilon
    )
    E_k = float(exp_config.simulation.mu)
    two_stage_settings = get_two_stage_settings(exp_config)
    stage_1_plan = load_privacy_matched_simulation_plan(
        config,
        stage=1,
    )
    stage_1_plan = map_stage_1_plan_runs(
        stage_1_plan,
        exp_config.hp_configuration_ids,
    )
    stage_1_plan = record_stage_1_run_scores(
        stage_1_plan,
        exp_config,
        evaluation_interval=evaluation_interval,
        simulations_root=(
            get_privacy_matched_simulations_directory(
                config=config,
                method=method,
                target_epsilon=target_epsilon,
                E_k=E_k,
            )
        ),
    )
    stage_1_plan = select_top_m_stage_1_runs(
        stage_1_plan,
        m=two_stage_settings.num_survivors,
        mode=stage_1_plan["evaluation"]["selection_mode"],
        seed=exp_config.seed,
    )
    return generate_stage_2_plan(
        stage_1_plan,
        exp_config,
        plan_directory=get_privacy_matched_plan_directory(
            config=config,
            method=method,
            target_epsilon=target_epsilon,
            E_k=E_k,
        ),
    )
