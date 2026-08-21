"""Result metadata shared by federated and centralized HPO experiments."""
import csv
import json
import math
from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

from .planning import (
    PAPERNOT_METHODS,
    PLAN_FILENAMES,
    TWO_STAGE_METHODS,
    generate_stage_2_plan,
    get_privacy_matched_plan_directory,
    get_privacy_matched_simulations_directory,
    get_privacy_matched_value_slug,
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

COMPUTE_MATCHED_RESULT_FILENAMES = {
    **RESULT_FILENAMES,
    "papernot_poisson_baseline": (
        "papernot_poisson_baseline_results.JSON"
    ),
    "two_stage_poisson_tuning": (
        "two_stage_poisson_tuning_results.JSON"
    ),
}

PRIVACY_MATCHED_RESULT_FILENAMES = {
    **COMPUTE_MATCHED_RESULT_FILENAMES,
}

COMPUTE_MATCHED_METHOD_LABELS = {
    "papernot_baseline": "Papernot--Steinke (TNB)",
    "papernot_poisson_baseline": "Papernot--Steinke (Poisson)",
    "two_stage_tuning": "Two-stage tuning (TNB)",
    "two_stage_poisson_tuning": "Two-stage tuning (Poisson)",
}

PRIVACY_MATCHED_METHOD_LABELS = {
    **COMPUTE_MATCHED_METHOD_LABELS,
}

def get_compute_matched_methods(config):
    """Return configured methods, preserving the legacy two-method default."""
    comparison = config.get("comparison", {})
    configured = comparison.get("methods")
    if configured is None:
        methods = list(RESULT_FILENAMES)
    else:
        methods = [str(method) for method in configured]
    if not methods:
        raise ValueError("comparison.methods must not be empty.")
    if len(set(methods)) != len(methods):
        raise ValueError("comparison.methods contains duplicates.")
    unknown = sorted(
        set(methods) - set(COMPUTE_MATCHED_RESULT_FILENAMES)
    )
    if unknown:
        raise ValueError(
            "Unknown compute-matched comparison method(s): "
            + ", ".join(unknown)
        )
    if "two_stage_tuning" not in methods:
        raise ValueError(
            "Compute matching requires two_stage_tuning as the reference "
            "method."
        )
    return methods


def get_privacy_matched_compilation_methods(exp_config):
    """Return an explicit method set, defaulting to the legacy pair."""
    configured = exp_config.compilation.get("methods")
    if configured is None:
        methods = ["papernot_baseline", "two_stage_tuning"]
    else:
        methods = [str(method) for method in configured]
    if not methods:
        raise ValueError("experiment.compilation.methods must not be empty.")
    if len(set(methods)) != len(methods):
        raise ValueError(
            "experiment.compilation.methods contains duplicates."
        )
    unknown = sorted(set(methods) - set(PRIVACY_MATCHED_METHOD_LABELS))
    if unknown:
        raise ValueError(
            "Unknown privacy-matched compilation method(s): "
            + ", ".join(unknown)
        )
    return methods


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

def load_compilation_plan(
    config,
    method,
    *,
    selection_signature=None,
):
    if method not in COMPUTE_MATCHED_RESULT_FILENAMES:
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
    if selection_signature is not None:
        if plan.get("plan_type") != "compute_matched":
            raise ValueError(
                f"Compilation plan {plan_filename} must have "
                "plan_type='compute_matched'. Regenerate the plan."
            )
        if plan.get("selection_signature") != selection_signature:
            raise ValueError(
                f"Compilation plan {plan_filename} has an incompatible "
                "selection signature. Regenerate the plan."
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
    """Fail once with all missing metric artifacts for configured methods."""
    simulations_root = get_compilation_paths(config)["simulations_root"]
    missing_paths = set()
    for method, plan in plans.items():
        execution_summary = plan.get("execution_summary")
        if not isinstance(execution_summary, dict):
            raise ValueError(
                f"The {method} compilation plan has no execution summary."
            )
        specs = execution_summary.get("required_stage_2_run_specs")
        if not isinstance(specs, list):
            raise ValueError(
                f"The {method} compilation plan has no required "
                "Stage-2 run specifications."
            )
        if not specs and method != "papernot_poisson_baseline":
            raise ValueError(
                f"The {method} compilation plan has an empty required "
                "Stage-2 run specification list."
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
            "Cannot compile the configured HPO methods because "
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
        ).strip().lower()

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
    if utility_at not in {"selection_round", "selected_checkpoint"}:
        raise ValueError(
            "evaluation.utility.at must be 'selection_round' or "
            f"'selected_checkpoint'; got {utility_at!r}."
        )
    if utility_at == "selected_checkpoint" and any(
        metric not in {"test_loss", "test_accuracy"}
        for metric in utility_metrics
    ):
        raise ValueError(
            "Selected-checkpoint utility metrics must be test_loss "
            "and/or test_accuracy."
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
        "uncertainty": {
            "scope": "conditional_on_reusable_trajectory_pool",
            "includes_base_training_variability": False,
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


def _load_and_validate_peak_artifact(
    run_path,
    stage,
    selection,
):
    """Validate that a trajectory score names a stored peak model."""
    run_path = Path(run_path)
    checkpoint_path = run_path / f"stage_{stage}_peak.pt"
    metadata_path = run_path / f"stage_{stage}_peak.JSON"
    missing_paths = [
        path
        for path in (checkpoint_path, metadata_path)
        if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Peak-aware compilation requires the selected model artifact "
            "and metadata; missing: "
            + ", ".join(str(path) for path in missing_paths)
        )
    try:
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Peak metadata is not valid JSON: {metadata_path}"
        ) from error
    if metadata.get("artifact_type") != "selected_peak_model":
        raise ValueError(
            f"Peak metadata has an invalid artifact type: {metadata_path}."
        )
    if metadata.get("schema_version") != 1:
        raise ValueError(
            f"Peak metadata has an unsupported schema: {metadata_path}."
        )
    observed_selection = metadata.get("selection")
    if not isinstance(observed_selection, dict):
        raise ValueError(
            f"Peak metadata has no selection object: {metadata_path}."
        )
    for field in ("metric", "mode", "stage", "round"):
        if observed_selection.get(field) != selection.get(field):
            raise ValueError(
                f"Peak metadata {metadata_path} has {field}="
                f"{observed_selection.get(field)!r}, expected "
                f"{selection.get(field)!r}."
            )
    if not math.isclose(
        float(observed_selection.get("score")),
        float(selection.get("score")),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Peak metadata score does not match metrics: {metadata_path}."
        )
    return {
        "release_checkpoint_path": str(checkpoint_path),
        "release_metadata_path": str(metadata_path),
        "release_stage": int(stage),
        "release_round": int(selection["round"]),
    }


def _attach_selected_checkpoint_utility(
    selected_run,
    evaluation,
    utility_evaluator,
):
    if evaluation["utility_at"] != "selected_checkpoint":
        return selected_run
    if utility_evaluator is None:
        raise ValueError(
            "Selected-checkpoint compilation requires a held-out utility "
            "evaluator."
        )
    utility = utility_evaluator(selected_run)
    if not isinstance(utility, dict):
        raise ValueError(
            "The held-out utility evaluator must return a mapping."
        )
    missing_metrics = [
        metric
        for metric in evaluation["utility_metrics"]
        if metric not in utility
    ]
    if missing_metrics:
        raise ValueError(
            "The held-out utility evaluator did not return metrics: "
            f"{missing_metrics}."
        )
    selected_run["utility"] = {
        metric: float(utility[metric])
        for metric in evaluation["utility_metrics"]
    }
    if any(
        not np.isfinite(value)
        for value in selected_run["utility"].values()
    ):
        raise ValueError(
            "The held-out utility evaluator returned a non-finite value."
        )
    selected_run["utility_source"] = "heldout_test_checkpoint_evaluation"
    return selected_run


def score_stage_peak_run(
    run,
    simulations_root,
    evaluation,
    stage,
    expected_start_round,
    expected_final_round,
    score_cache,
    evaluation_interval=1,
):
    """Score and validate the selected peak of exactly one stage."""
    directory_key = f"stage_{stage}_run_directory"
    run_directory = str(run[directory_key])
    cache_key = (
        "stage_peak",
        int(stage),
        run_directory,
        evaluation["selection_metric"],
        evaluation["evaluation_mode"],
        int(evaluation_interval),
    )
    if cache_key not in score_cache:
        run_path = Path(simulations_root) / run_directory
        metrics_path = run_path / f"stage_{stage}.csv"
        trajectory_evaluation = evaluate_metric_trajectory(
            metric_segments=[
                {
                    "stage": int(stage),
                    "csv_path": metrics_path,
                    "expected_start_round": int(expected_start_round),
                    "expected_final_round": int(expected_final_round),
                    "evaluation_interval": int(evaluation_interval),
                }
            ],
            selection_metric=evaluation["selection_metric"],
            evaluation_mode=evaluation["evaluation_mode"],
            utility_metrics=[],
        )
        selection = trajectory_evaluation["selection"]
        release_metadata = _load_and_validate_peak_artifact(
            run_path=run_path,
            stage=stage,
            selection=selection,
        )
        score_cache[cache_key] = {
            "selection": selection,
            "utility": {},
            f"stage_{stage}_metrics_path": str(metrics_path),
            **release_metadata,
        }
    scored_run = copy.deepcopy(run)
    scored_run.update(copy.deepcopy(score_cache[cache_key]))
    return scored_run

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
        evaluation["utility_at"],
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
        trajectory_utility_metrics = (
            []
            if evaluation["utility_at"] == "selected_checkpoint"
            else evaluation["utility_metrics"]
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
            utility_metrics=trajectory_utility_metrics,
        )
        cached_score = {
            "selection": trajectory_evaluation["selection"],
            "utility": trajectory_evaluation["utility"],
            "stage_1_metrics_path": str(stage_1_metrics_path),
            "stage_2_metrics_path": str(stage_2_metrics_path),
        }
        if evaluation["utility_at"] == "selected_checkpoint":
            release_stage = int(
                trajectory_evaluation["selection"]["stage"]
            )
            release_run_path = (
                simulations_root / stage_1_directory
                if release_stage == 1
                else simulations_root / stage_2_directory
            )
            cached_score.update(
                _load_and_validate_peak_artifact(
                    run_path=release_run_path,
                    stage=release_stage,
                    selection=trajectory_evaluation["selection"],
                )
            )
        score_cache[cache_key] = cached_score

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


def summarize_sampled_k(trials):
    """Summarize realized random-search breadth without dropping K=0."""
    sampled_k = np.asarray(
        [int(trial["sampled_K"]) for trial in trials],
        dtype=int,
    )
    if sampled_k.size == 0 or np.any(sampled_k < 0):
        raise ValueError(
            "Sampled-K summaries require nonnegative counts and at least "
            "one trial."
        )
    return {
        "num_trials": int(sampled_k.size),
        "mean": float(np.mean(sampled_k)),
        "median": float(np.median(sampled_k)),
        "minimum": int(np.min(sampled_k)),
        "maximum": int(np.max(sampled_k)),
        "num_K_zero": int(np.count_nonzero(sampled_k == 0)),
        "rate_K_zero": float(np.mean(sampled_k == 0)),
        "num_K_one": int(np.count_nonzero(sampled_k == 1)),
        "rate_K_one": float(np.mean(sampled_k == 1)),
        "num_K_le_two": int(np.count_nonzero(sampled_k <= 2)),
        "rate_K_le_two": float(np.mean(sampled_k <= 2)),
    }


def compile_papernot_results(
    config,
    plan,
    evaluation,
    stage_compute_schedule,
    evaluation_interval=1,
    simulations_root=None,
    utility_evaluator=None,
):
    result = copy.deepcopy(plan)
    result["result_type"] = "compiled_hpo_results"
    result["evaluation"] = get_compiled_evaluation_metadata(
        evaluation=evaluation,
        stage_1_end=int(config.simulation.stage_1_end),
        stage_2_end=int(config.simulation.stage_2_end),
        evaluation_interval=evaluation_interval,
    )
    method = str(result.get("method"))
    if method not in PAPERNOT_METHODS:
        raise ValueError(
            f"Papernot compilation received method {method!r}."
        )
    if evaluation["utility_at"] == "selected_checkpoint":
        result["final_candidate_policy"] = (
            "one best validation checkpoint across Stage 1 and Stage 2 "
            "for each complete Papernot base run, followed by top-1 "
            "selection across sampled base runs; for Poisson K=0, return "
            "the plan's fixed data-independent random model"
        )
    if simulations_root is None:
        simulations_root = get_compilation_paths(config)[
            "simulations_root"
        ]
    else:
        simulations_root = Path(simulations_root)
    stage_1_end = int(config.simulation.stage_1_end)
    stage_2_end = int(config.simulation.stage_2_end)
    score_cache = {}

    for point_index, point in enumerate(result["points"]):
        point["expected_compute"] = float(
            float(point["E_K"]) * sum(stage_compute_schedule)
        )
        for trial_index, trial in enumerate(point["trials"]):
            sampled_k = int(trial["sampled_K"])
            if sampled_k == 0:
                if method != "papernot_poisson_baseline":
                    raise ValueError(
                        "Only the Poisson Papernot mechanism permits K=0."
                    )
                if trial.get("sampled_stage_2_runs") != []:
                    raise ValueError(
                        "A Poisson K=0 trial must contain no trained runs."
                    )
                if utility_evaluator is None or not hasattr(
                    utility_evaluator,
                    "build_data_independent_fallback",
                ):
                    raise ValueError(
                        "Poisson K=0 compilation requires an evaluator "
                        "that can construct the data-independent fallback."
                    )
                fallback_definition = result.get("k_zero_fallback")
                if not isinstance(fallback_definition, dict):
                    raise ValueError(
                        "The Poisson plan has no K=0 fallback definition."
                    )
                trial["selection_tie_break_seed"] = None
                trial["final_selected_run"] = copy.deepcopy(
                    utility_evaluator.build_data_independent_fallback(
                        fallback_definition=fallback_definition,
                        evaluation=evaluation,
                    )
                )
                continue
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
            trial["final_selected_run"] = (
                _attach_selected_checkpoint_utility(
                    selected_run=selected_runs[0],
                    evaluation=evaluation,
                    utility_evaluator=utility_evaluator,
                )
            )

        point["aggregate_utility"] = (
            summarize_trial_utilities(
                trials=point["trials"],
                utility_metrics=evaluation["utility_metrics"],
            )
        )
        point["sampled_k_summary"] = summarize_sampled_k(
            point["trials"]
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
    method_labels = COMPUTE_MATCHED_METHOD_LABELS
    method_styles = {
        "papernot_baseline": {"linestyle": "-", "marker": "o"},
        "papernot_poisson_baseline": {
            "linestyle": ":",
            "marker": "^",
        },
        "two_stage_tuning": {"linestyle": "--", "marker": "s"},
        "two_stage_poisson_tuning": {
            "linestyle": "-.",
            "marker": "D",
        },
    }
    methods = list(dict.fromkeys(row["method"] for row in rows))
    utility_metrics = sorted(
        {row["utility_metric"] for row in rows}
    )
    plot_paths = []

    for utility_metric in utility_metrics:
        figure, axis = plt.subplots(figsize=(7, 5))
        for method in methods:
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
                **method_styles[method],
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
                if method in PAPERNOT_METHODS:
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
                            "sampling_distribution": result.get(
                                "sampling_distribution",
                                "tnb",
                            ),
                            "sampled_K_is_zero": (
                                int(stage_1_sampled_K == 0)
                            ),
                            "sampled_K_is_one": (
                                int(stage_1_sampled_K == 1)
                            ),
                            "sampled_K_le_two": (
                                int(stage_1_sampled_K <= 2)
                            ),
                            "used_K_zero_fallback": int(
                                selected_run.get("candidate_origin")
                                == (
                                    "poisson_k_zero_"
                                    "data_independent_fallback"
                                )
                            ),
                            "hp_configuration_id": selected_run[
                                "hp_configuration_id"
                            ],
                            "stage_1_run_index": selected_run[
                                "stage_1_run_index"
                            ],
                            "continuation_index": selected_run.get(
                                "continuation_index",
                                "",
                            ),
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
                            "utility_source": selected_run.get(
                                "utility_source",
                                "trajectory_metrics_csv",
                            ),
                            "candidate_origin": selected_run.get(
                                "candidate_origin",
                                "complete_trajectory",
                            ),
                            "release_stage": selected_run.get(
                                "release_stage",
                                selected_run["selection"]["stage"],
                            ),
                            "release_round": selected_run.get(
                                "release_round",
                                selected_run["selection"]["round"],
                            ),
                            "release_checkpoint_path": selected_run.get(
                                "release_checkpoint_path",
                                "",
                            ),
                            "release_metadata_path": selected_run.get(
                                "release_metadata_path",
                                "",
                            ),
                            "stage_1_metrics_path": selected_run.get(
                                "stage_1_metrics_path",
                                "",
                            ),
                            "stage_2_metrics_path": selected_run.get(
                                "stage_2_metrics_path",
                                "",
                            ),
                            "uncertainty_scope": (
                                "conditional_on_reusable_trajectory_pool"
                            ),
                        }
                    )
    return rows

def compile_two_stage_results(
    config,
    plan,
    evaluation,
    stage_compute_schedule,
    evaluation_interval=1,
    simulations_root=None,
    utility_evaluator=None,
):
    result = copy.deepcopy(plan)
    result["result_type"] = "compiled_hpo_results"
    result["evaluation"] = get_compiled_evaluation_metadata(
        evaluation=evaluation,
        stage_1_end=int(config.simulation.stage_1_end),
        stage_2_end=int(config.simulation.stage_2_end),
        evaluation_interval=evaluation_interval,
    )
    if evaluation["utility_at"] == "selected_checkpoint":
        result["final_candidate_policy"] = (
            "all Stage-1 survivor peak checkpoints plus all sampled "
            "Stage-2 peak checkpoints, followed by top-1 validation "
            "selection"
        )
    if simulations_root is None:
        simulations_root = get_compilation_paths(config)[
            "simulations_root"
        ]
    else:
        simulations_root = Path(simulations_root)
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
            scored_stage_2_runs = []
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
                if evaluation["utility_at"] == "selected_checkpoint":
                    scored_run = score_stage_peak_run(
                        run=complete_run,
                        simulations_root=simulations_root,
                        evaluation=evaluation,
                        stage=2,
                        expected_start_round=stage_1_end,
                        expected_final_round=stage_2_end - 1,
                        score_cache=score_cache,
                        evaluation_interval=evaluation_interval,
                    )
                    scored_run["candidate_origin"] = "stage_2_peak"
                else:
                    scored_run = score_complete_run(
                        run=complete_run,
                        simulations_root=simulations_root,
                        evaluation=evaluation,
                        stage_1_end=stage_1_end,
                        stage_2_end=stage_2_end,
                        score_cache=score_cache,
                        evaluation_interval=evaluation_interval,
                    )
                scored_stage_2_runs.append(scored_run)

            stage_2_trial["sampled_stage_2_runs"] = (
                scored_stage_2_runs
            )
            if evaluation["utility_at"] == "selected_checkpoint":
                scored_stage_1_survivors = []
                for survivor in trial["trial_stage_1"][
                    "top_m_stage_1_runs"
                ]:
                    scored_survivor = score_stage_peak_run(
                        run=survivor,
                        simulations_root=simulations_root,
                        evaluation=evaluation,
                        stage=1,
                        expected_start_round=0,
                        expected_final_round=stage_1_end - 1,
                        score_cache=score_cache,
                        evaluation_interval=evaluation_interval,
                    )
                    scored_survivor["candidate_origin"] = (
                        "stage_1_survivor_peak"
                    )
                    scored_stage_1_survivors.append(scored_survivor)
                trial["stage_1_final_candidates"] = (
                    scored_stage_1_survivors
                )
                scored_runs = [
                    *scored_stage_1_survivors,
                    *scored_stage_2_runs,
                ]
            else:
                scored_runs = scored_stage_2_runs
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
            trial["final_selected_run"] = (
                _attach_selected_checkpoint_utility(
                    selected_run=selected_runs[0],
                    evaluation=evaluation,
                    utility_evaluator=utility_evaluator,
                )
            )

        point["aggregate_utility"] = (
            summarize_trial_utilities(
                trials=point["trials"],
                utility_metrics=evaluation["utility_metrics"],
            )
        )
        point["sampled_k_summary"] = summarize_sampled_k(
            [trial["trial_stage_1"] for trial in point["trials"]]
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
    utility_evaluator=None,
    selection_signature=None,
):
    """Compile configured compute-matched methods and persist artifacts."""
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
    methods = get_compute_matched_methods(config)
    paths = get_compilation_paths(config)
    plans = {
        method: load_compilation_plan(
            config,
            method,
            selection_signature=selection_signature,
        )
        for method in methods
    }
    validate_compilation_metric_files(config, plans)
    compiled_results = {}
    for method in methods:
        compiler = (
            compile_papernot_results
            if method in PAPERNOT_METHODS
            else compile_two_stage_results
        )
        compiled_results[method] = compiler(
            config=config,
            plan=plans[method],
            evaluation=evaluation,
            stage_compute_schedule=stage_compute_schedule,
            evaluation_interval=evaluation_interval,
            utility_evaluator=utility_evaluator,
        )

    two_stage_compute = np.asarray(
        [
            point["expected_compute"]
            for point in compiled_results["two_stage_tuning"]["points"]
        ],
        dtype=float,
    )
    for method, result in compiled_results.items():
        method_compute = np.asarray(
            [point["expected_compute"] for point in result["points"]],
            dtype=float,
        )
        if (
            method_compute.shape != two_stage_compute.shape
            or not np.allclose(
                method_compute,
                two_stage_compute,
                rtol=1e-12,
                atol=1e-9,
            )
        ):
            raise ValueError(
                f"{method} and two_stage_tuning do not have matching "
                "expected-compute coordinates. Regenerate the plans."
            )

    compiled_root = paths["compiled_root"]
    compiled_root.mkdir(parents=True, exist_ok=True)
    result_paths = {}
    for method, result in compiled_results.items():
        plan_path = paths["plan_root"] / PLAN_FILENAMES[method][2]
        result["source_plan_path"] = str(plan_path)
        result_path = (
            compiled_root / COMPUTE_MATCHED_RESULT_FILENAMES[method]
        )
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
    if evaluation["utility_at"] == "selected_checkpoint":
        selection_signature = stage_1_plan.get("selection_signature")
        if not isinstance(selection_signature, dict):
            raise ValueError(
                "Peak-aware Stage-2 planning requires a Stage-1 plan "
                "selection_signature. Regenerate the Stage-1 plan."
            )
        stage_1_plan["evaluation"]["selection_signature"] = copy.deepcopy(
            selection_signature
        )

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
                if evaluation["utility_at"] == "selected_checkpoint":
                    peak_selection = {
                        "metric": metric,
                        "mode": evaluation_mode,
                        "selection_mode": evaluation["selection_mode"],
                        "stage": 1,
                        "round": int(
                            sampled_run["evaluation_round"]
                        ),
                        "score": float(
                            sampled_run["evaluation_score"]
                        ),
                    }
                    sampled_run.update(
                        _load_and_validate_peak_artifact(
                            run_path=(
                                simulations_root / run_directory
                            ),
                            stage=1,
                            selection=peak_selection,
                        )
                    )

    return stage_1_plan


def generate_stage_2_plan_from_results(
    config,
    *,
    evaluation_interval=1,
    selection_signature=None,
):
    """Score Stage-1 runs, select top-m, and persist the Stage-2 plan."""
    method = str(config.simulation.method)
    if method not in TWO_STAGE_METHODS:
        raise ValueError(
            "Stage-2 plan generation requires "
            "a recognized two-stage simulation method."
        )

    two_stage_settings = get_two_stage_settings(config)
    stage_1_plan = load_simulation_plan(
        config,
        stage=1,
        selection_signature=selection_signature,
    )
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
    return generate_stage_2_plan(
        stage_1_plan,
        config,
        plan_filename=PLAN_FILENAMES[method][2],
    )


def generate_privacy_matched_stage_2_plan_from_results(
    config,
    *,
    evaluation_interval=1,
):
    """Generate one nested Stage-2 plan from a completed Stage 1."""
    exp_config = config.experiment
    method = str(exp_config.simulation.method)
    if method not in TWO_STAGE_METHODS:
        raise ValueError(
            "Privacy-matched Stage-2 plan generation requires "
            "a recognized two-stage simulation method."
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
        plan_filename=PLAN_FILENAMES[method][2],
        plan_directory=get_privacy_matched_plan_directory(
            config=config,
            method=method,
            target_epsilon=target_epsilon,
            E_k=E_k,
        ),
    )


def _privacy_matched_compilation_config(
    config,
    *,
    method,
    target_epsilon,
    E_k,
):
    """Copy a config and select one privacy-matched plan cell."""
    cell_config = OmegaConf.create(
        OmegaConf.to_container(config, resolve=True)
    )
    cell_config.experiment.simulation.method = str(method)
    cell_config.experiment.simulation.target_epsilon = float(
        target_epsilon
    )
    cell_config.experiment.simulation.mu = float(E_k)
    return cell_config


def compile_privacy_matched_sigma_privacy(config):
    """Compile calibrated sigma versus target epsilon from Stage-1 plans."""
    exp_config = config.experiment
    try:
        target_epsilons = [
            float(value)
            for value in exp_config.compilation.target_epsilons
        ]
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "experiment.compilation.target_epsilons must contain "
            "the privacy targets to compile."
        ) from error
    if not target_epsilons:
        raise ValueError(
            "experiment.compilation.target_epsilons must not be empty."
        )
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in target_epsilons
    ):
        raise ValueError(
            "Every compilation target epsilon must be finite and positive."
        )
    if len(set(target_epsilons)) != len(target_epsilons):
        raise ValueError(
            "experiment.compilation.target_epsilons contains duplicates."
        )
    target_epsilons.sort()

    E_k = float(exp_config.simulation.mu)
    if not math.isfinite(E_k) or E_k <= 0.0:
        raise ValueError("experiment.simulation.mu must be positive.")
    methods = get_privacy_matched_compilation_methods(exp_config)

    rows = []
    for method in methods:
        method_rows = []
        for target_epsilon in target_epsilons:
            cell_config = _privacy_matched_compilation_config(
                config,
                method=method,
                target_epsilon=target_epsilon,
                E_k=E_k,
            )
            plan = load_privacy_matched_simulation_plan(
                cell_config,
                stage=1,
            )
            point = plan["points"][0]
            plan_path = (
                get_privacy_matched_plan_directory(
                    config=cell_config,
                    method=method,
                    target_epsilon=target_epsilon,
                    E_k=E_k,
                )
                / PLAN_FILENAMES[method][1]
            )
            row = {
                "method": method,
                "method_label": PRIVACY_MATCHED_METHOD_LABELS[method],
                "target_epsilon": float(point["target_epsilon"]),
                "achieved_epsilon": float(point["achieved_epsilon"]),
                "noise_multiplier": float(point["noise_multiplier"]),
                "mu": E_k,
                "delta": float(point["delta"]),
                "best_renyi_order": float(point["best_renyi_order"]),
                "source_plan_path": str(plan_path),
                "sampling_distribution": plan.get(
                    "sampling_distribution",
                    "tnb",
                ),
                "probability_K_zero": point.get(
                    "probability_K_zero",
                    "",
                ),
            }
            method_rows.append(row)
            rows.append(row)

        sigmas = [row["noise_multiplier"] for row in method_rows]
        if any(
            next_sigma > sigma + 1e-12
            for sigma, next_sigma in zip(sigmas, sigmas[1:])
        ):
            raise ValueError(
                f"Calibrated sigma for {method} increases as target "
                "epsilon increases. Check that the plans use consistent "
                "privacy-accounting settings."
            )

    output_directory = (
        Path(exp_config.output.results_root)
        / str(exp_config.name)
        / str(exp_config.run_id)
        / "compiled_result"
        / f"mu_{get_privacy_matched_value_slug(E_k, 'mu')}"
        / "sigma_privacy_plot"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    csv_path = output_directory / "sigma_vs_epsilon.csv"
    save_trial_result_rows(rows=rows, csv_path=csv_path)

    metadata_path = output_directory / "sigma_vs_epsilon.JSON"
    _save_json_atomically(
        {
            "schema_version": 1,
            "plot": "sigma_vs_epsilon",
            "run_id": str(exp_config.run_id),
            "mu": E_k,
            "target_epsilons": target_epsilons,
            "methods": methods,
            "rows": rows,
        },
        metadata_path,
    )

    figure, axis = plt.subplots(figsize=(7, 5))
    method_styles = {
        "papernot_baseline": {"linestyle": "-", "marker": "o"},
        "papernot_poisson_baseline": {
            "linestyle": ":",
            "marker": "^",
        },
        "two_stage_tuning": {"linestyle": "--", "marker": "s"},
        "two_stage_poisson_tuning": {
            "linestyle": "-.",
            "marker": "D",
        },
    }
    for method in methods:
        method_label = PRIVACY_MATCHED_METHOD_LABELS[method]
        method_rows = [row for row in rows if row["method"] == method]
        axis.plot(
            [row["target_epsilon"] for row in method_rows],
            [row["noise_multiplier"] for row in method_rows],
            **method_styles[method],
            label=method_label,
        )
    axis.set_xlabel(r"Target privacy budget $\epsilon$")
    axis.set_ylabel(r"Noise multiplier $\sigma$")
    axis.set_title(rf"Noise Calibration at $\mathbb{{E}}[K]={E_k:g}$")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    plot_path = output_directory / "sigma_vs_epsilon.png"
    figure.savefig(plot_path, dpi=300)
    plt.close(figure)

    return {
        "output_directory": output_directory,
        "csv_path": csv_path,
        "metadata_path": metadata_path,
        "plot_path": plot_path,
        "rows": rows,
    }


def compile_privacy_matched_expected_compute_privacy(config):
    """Compile expected compute versus epsilon for configured mu values."""
    exp_config = config.experiment
    try:
        target_epsilons = sorted(
            float(value)
            for value in exp_config.compilation.target_epsilons
        )
        target_mus = sorted(
            float(value)
            for value in exp_config.compilation.target_mus
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "experiment.compilation.target_epsilons and target_mus "
            "must contain the coordinates to compile."
        ) from error
    for values, name in (
        (target_epsilons, "target_epsilons"),
        (target_mus, "target_mus"),
    ):
        if not values:
            raise ValueError(
                f"experiment.compilation.{name} must not be empty."
            )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in values
        ):
            raise ValueError(
                f"Every experiment.compilation.{name} value must be "
                "finite and positive."
            )
        if len(set(values)) != len(values):
            raise ValueError(
                f"experiment.compilation.{name} contains duplicates."
            )

    stage_1_compute = int(exp_config.simulation.stage_1_end)
    stage_2_end = int(exp_config.simulation.stage_2_end)
    stage_2_compute = stage_2_end - stage_1_compute
    if stage_1_compute <= 0 or stage_2_compute <= 0:
        raise ValueError(
            "Expected-compute compilation requires 0 < stage_1_end "
            "< stage_2_end."
        )
    two_stage_settings = get_two_stage_settings(exp_config)
    stage_2_expected_trials = float(
        two_stage_settings.stage_2_expected_trials
    )
    methods = get_privacy_matched_compilation_methods(exp_config)

    rows = []
    for E_k in target_mus:
        expected_compute_by_method = {
            "papernot_baseline": E_k * stage_2_end,
            "papernot_poisson_baseline": E_k * stage_2_end,
            "two_stage_tuning": (
                E_k * stage_1_compute
                + stage_2_expected_trials * stage_2_compute
            ),
            "two_stage_poisson_tuning": (
                E_k * stage_1_compute
                + stage_2_expected_trials * stage_2_compute
            ),
        }
        for method in methods:
            for target_epsilon in target_epsilons:
                rows.append(
                    {
                        "method": method,
                        "method_label": (
                            PRIVACY_MATCHED_METHOD_LABELS[method]
                        ),
                        "target_epsilon": target_epsilon,
                        "mu": E_k,
                        "expected_compute": float(
                            expected_compute_by_method[method]
                        ),
                        "stage_1_compute_per_run": stage_1_compute,
                        "stage_2_compute_per_run": stage_2_compute,
                        "stage_1_expected_trials": (
                            E_k
                            if method in TWO_STAGE_METHODS
                            else ""
                        ),
                        "stage_2_expected_trials": (
                            stage_2_expected_trials
                            if method in TWO_STAGE_METHODS
                            else ""
                        ),
                        "full_run_expected_trials": (
                            E_k
                            if method in PAPERNOT_METHODS
                            else ""
                        ),
                    }
                )

    output_directory = (
        Path(exp_config.output.results_root)
        / str(exp_config.name)
        / str(exp_config.run_id)
        / "compiled_result"
        / "expected_compute_privacy_plot"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    csv_path = output_directory / "expected_compute_vs_epsilon.csv"
    save_trial_result_rows(rows=rows, csv_path=csv_path)
    metadata_path = output_directory / "expected_compute_vs_epsilon.JSON"
    _save_json_atomically(
        {
            "schema_version": 1,
            "plot": "expected_compute_vs_epsilon",
            "data_source": "experiment_configuration",
            "run_id": str(exp_config.run_id),
            "target_epsilons": target_epsilons,
            "target_mus": target_mus,
            "stage_1_compute_per_run": stage_1_compute,
            "stage_2_compute_per_run": stage_2_compute,
            "stage_2_expected_trials": stage_2_expected_trials,
            "methods": methods,
            "rows": rows,
        },
        metadata_path,
    )

    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    color_map = plt.get_cmap("tab10")
    method_styles = {
        "papernot_baseline": {"linestyle": "-", "marker": "o"},
        "papernot_poisson_baseline": {
            "linestyle": ":",
            "marker": "^",
        },
        "two_stage_tuning": {"linestyle": "--", "marker": "s"},
        "two_stage_poisson_tuning": {
            "linestyle": "-.",
            "marker": "D",
        },
    }
    for mu_index, E_k in enumerate(target_mus):
        color = color_map(mu_index % color_map.N)
        for method in methods:
            plot_rows = [
                row
                for row in rows
                if row["method"] == method
                and math.isclose(
                    float(row["mu"]),
                    E_k,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ]
            axis.plot(
                [row["target_epsilon"] for row in plot_rows],
                [row["expected_compute"] for row in plot_rows],
                color=color,
                **method_styles[method],
            )

    method_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            label=method_label,
            **method_styles[method],
        )
        for method in methods
        for method_label in [PRIVACY_MATCHED_METHOD_LABELS[method]]
    ]
    mu_handles = [
        Line2D(
            [0],
            [0],
            color=color_map(index % color_map.N),
            linewidth=3,
            label=rf"$\mu={E_k:g}$",
        )
        for index, E_k in enumerate(target_mus)
    ]
    axis.legend(
        handles=method_handles + mu_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=min(4, len(method_handles) + len(mu_handles)),
    )
    axis.set_xticks(target_epsilons)
    axis.set_xlabel(r"Target privacy budget $\epsilon$")
    axis.set_ylabel("Expected compute (optimizer updates)")
    axis.set_title("Expected Compute Across Privacy Budgets")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    plot_path = output_directory / "expected_compute_vs_epsilon.png"
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return {
        "output_directory": output_directory,
        "csv_path": csv_path,
        "metadata_path": metadata_path,
        "plot_path": plot_path,
        "rows": rows,
    }


def plot_privacy_matched_utility(rows, output_directory):
    """Plot trial-mean utility against target epsilon with 95% CIs."""
    utility_metrics = list(
        dict.fromkeys(row["utility_metric"] for row in rows)
    )
    plot_paths = []
    plotted_methods = list(
        dict.fromkeys(row["method"] for row in rows)
    )
    method_styles = {
        "papernot_baseline": {"linestyle": "-", "marker": "o"},
        "papernot_poisson_baseline": {
            "linestyle": ":",
            "marker": "^",
        },
        "two_stage_tuning": {"linestyle": "--", "marker": "s"},
        "two_stage_poisson_tuning": {
            "linestyle": "-.",
            "marker": "D",
        },
    }
    for utility_metric in utility_metrics:
        figure, axis = plt.subplots(figsize=(7, 5))
        for method in plotted_methods:
            method_label = PRIVACY_MATCHED_METHOD_LABELS[method]
            method_rows = [
                row
                for row in rows
                if row["method"] == method
                and row["utility_metric"] == utility_metric
            ]
            values_by_epsilon = {}
            for row in method_rows:
                values_by_epsilon.setdefault(
                    float(row["target_epsilon"]),
                    [],
                ).append(float(row["utility_score"]))

            target_epsilons = sorted(values_by_epsilon)
            means = []
            ci95_half_widths = []
            for target_epsilon in target_epsilons:
                values = np.asarray(
                    values_by_epsilon[target_epsilon],
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
                target_epsilons,
                means,
                **method_styles[method],
                label=method_label,
            )
            axis.fill_between(
                target_epsilons,
                means - ci95_half_widths,
                means + ci95_half_widths,
                color=mean_line.get_color(),
                alpha=0.2,
                linewidth=0,
            )

        metric_label = utility_metric.replace("_", " ").title()
        axis.set_xlabel(r"Target privacy budget $\epsilon$")
        axis.set_ylabel(metric_label)
        axis.set_title(f"{metric_label} vs Privacy Budget")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        plot_path = output_directory / f"{utility_metric}_vs_epsilon.png"
        figure.savefig(plot_path, dpi=300)
        plt.close(figure)
        plot_paths.append(plot_path)

    return plot_paths


def compile_privacy_matched_utility_privacy(
    config,
    *,
    utility_evaluator=None,
):
    """Compile trial utility versus epsilon for one configured mu."""
    exp_config = config.experiment
    try:
        target_epsilons = sorted(
            float(value)
            for value in exp_config.compilation.target_epsilons
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "experiment.compilation.target_epsilons must contain "
            "the privacy targets to compile."
        ) from error
    if not target_epsilons:
        raise ValueError(
            "experiment.compilation.target_epsilons must not be empty."
        )
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in target_epsilons
    ):
        raise ValueError(
            "Every compilation target epsilon must be finite and positive."
        )
    if len(set(target_epsilons)) != len(target_epsilons):
        raise ValueError(
            "experiment.compilation.target_epsilons contains duplicates."
        )

    E_k = float(exp_config.simulation.mu)
    if not math.isfinite(E_k) or E_k <= 0.0:
        raise ValueError("experiment.simulation.mu must be positive.")
    stage_1_end = int(exp_config.simulation.stage_1_end)
    stage_2_end = int(exp_config.simulation.stage_2_end)
    stage_compute_schedule = [
        stage_1_end,
        stage_2_end - stage_1_end,
    ]
    if any(value <= 0 for value in stage_compute_schedule):
        raise ValueError(
            "Utility compilation requires 0 < stage_1_end < stage_2_end."
        )
    evaluation_interval = int(config.run_settings.evaluation_interval)
    if evaluation_interval <= 0:
        raise ValueError(
            "run_settings.evaluation_interval must be positive."
        )
    evaluation = get_evaluation_settings(exp_config)
    methods = get_privacy_matched_compilation_methods(exp_config)

    compiled_results = {}
    source_plan_paths = {
        method: []
        for method in methods
    }
    trial_rows = []
    for method in methods:
        method_result = None
        for target_epsilon in target_epsilons:
            cell_config = _privacy_matched_compilation_config(
                config,
                method=method,
                target_epsilon=target_epsilon,
                E_k=E_k,
            )
            plan = load_privacy_matched_simulation_plan(
                cell_config,
                stage=2,
            )
            simulations_root = (
                get_privacy_matched_simulations_directory(
                    config=cell_config,
                    method=method,
                    target_epsilon=target_epsilon,
                    E_k=E_k,
                )
            )
            compiler = (
                compile_papernot_results
                if method in PAPERNOT_METHODS
                else compile_two_stage_results
            )
            cell_result = compiler(
                config=cell_config.experiment,
                plan=plan,
                evaluation=evaluation,
                stage_compute_schedule=stage_compute_schedule,
                evaluation_interval=evaluation_interval,
                simulations_root=simulations_root,
                utility_evaluator=utility_evaluator,
            )
            plan_path = (
                get_privacy_matched_plan_directory(
                    config=cell_config,
                    method=method,
                    target_epsilon=target_epsilon,
                    E_k=E_k,
                )
                / PLAN_FILENAMES[method][2]
            )
            source_plan_paths[method].append(str(plan_path))
            compiled_point = cell_result["points"][0]
            compiled_point["source_plan_path"] = str(plan_path)

            if method_result is None:
                method_result = {
                    key: value
                    for key, value in cell_result.items()
                    if key != "points"
                }
                method_result.update(
                    {
                        "schema_version": 1,
                        "mu": E_k,
                        "points": [],
                    }
                )
            method_result["points"].append(compiled_point)

            cell_rows = build_trial_result_rows(
                {method: cell_result}
            )
            for row in cell_rows:
                trial_rows.append(
                    {
                        "method": row["method"],
                        "target_epsilon": float(
                            compiled_point["target_epsilon"]
                        ),
                        "achieved_epsilon": float(
                            compiled_point["achieved_epsilon"]
                        ),
                        "noise_multiplier": float(
                            compiled_point["noise_multiplier"]
                        ),
                        "delta": float(compiled_point["delta"]),
                        "mu": E_k,
                        **{
                            key: value
                            for key, value in row.items()
                            if key != "method"
                        },
                    }
                )

        method_result["source_plan_paths"] = source_plan_paths[method]
        compiled_results[method] = method_result

    output_directory = (
        Path(exp_config.output.results_root)
        / str(exp_config.name)
        / str(exp_config.run_id)
        / "compiled_result"
        / f"mu_{get_privacy_matched_value_slug(E_k, 'mu')}"
        / "utility_privacy_plot"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    result_paths = {}
    for method, result in compiled_results.items():
        result_path = (
            output_directory
            / PRIVACY_MATCHED_RESULT_FILENAMES[method]
        )
        _save_json_atomically(result, result_path)
        result_paths[method] = result_path

    trial_csv_path = output_directory / "trial_results.csv"
    save_trial_result_rows(rows=trial_rows, csv_path=trial_csv_path)

    summary_rows = []
    for method, result in compiled_results.items():
        for point in result["points"]:
            for utility_metric, summary in point[
                "aggregate_utility"
            ].items():
                summary_rows.append(
                    {
                        "method": method,
                        "target_epsilon": float(
                            point["target_epsilon"]
                        ),
                        "achieved_epsilon": float(
                            point["achieved_epsilon"]
                        ),
                        "noise_multiplier": float(
                            point["noise_multiplier"]
                        ),
                        "delta": float(point["delta"]),
                        "mu": E_k,
                        "selection_metric": evaluation[
                            "selection_metric"
                        ],
                        "selection_mode": evaluation[
                            "selection_mode"
                        ],
                        "utility_metric": utility_metric,
                        **{
                            f"sampled_k_{key}": value
                            for key, value in point.get(
                                "sampled_k_summary",
                                {},
                            ).items()
                        },
                        **summary,
                    }
                )
    summary_csv_path = output_directory / "utility_summary.csv"
    save_trial_result_rows(
        rows=summary_rows,
        csv_path=summary_csv_path,
    )
    plot_paths = plot_privacy_matched_utility(
        rows=trial_rows,
        output_directory=output_directory,
    )

    return {
        "output_directory": output_directory,
        "result_paths": result_paths,
        "trial_csv_path": trial_csv_path,
        "summary_csv_path": summary_csv_path,
        "plot_paths": plot_paths,
    }


def compile_privacy_matched_results(
    config,
    *,
    utility_evaluator=None,
):
    """Run the configured privacy-matched compilation targets."""
    try:
        targets = [
            str(target)
            for target in config.experiment.compilation.targets
        ]
    except (AttributeError, TypeError) as error:
        raise ValueError(
            "experiment.compilation.targets must be a non-empty list."
        ) from error
    if not targets:
        raise ValueError(
            "experiment.compilation.targets must be a non-empty list."
        )
    if len(set(targets)) != len(targets):
        raise ValueError(
            "experiment.compilation.targets contains duplicates."
        )

    compilers = {
        "sigma_privacy": compile_privacy_matched_sigma_privacy,
        "expected_compute_privacy": (
            compile_privacy_matched_expected_compute_privacy
        ),
        "utility_privacy": compile_privacy_matched_utility_privacy,
    }
    unknown_targets = sorted(set(targets) - set(compilers))
    if unknown_targets:
        raise ValueError(
            "Unknown privacy-matched compilation target(s): "
            f"{', '.join(unknown_targets)}. Supported targets: "
            f"{', '.join(sorted(compilers))}."
        )
    outputs = {}
    for target in targets:
        if target == "utility_privacy":
            outputs[target] = compilers[target](
                config,
                utility_evaluator=utility_evaluator,
            )
        else:
            outputs[target] = compilers[target](config)
    return outputs
