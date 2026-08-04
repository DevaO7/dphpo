import numpy as np
import matplotlib.pyplot as plt
import hydra
import csv
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from privacy_accounting.tnb import _solve_gamma_for_conditional_mean, TNBDistribution
from pathlib import Path
import json
from collections import Counter
from utils.data_utils import get_data_loaders, set_seed
from flearn.trainmodel import models
from flearn.servers.server_avg import FedAvg
from privacy_accounting.dpfedavg import compute_dpfedavg_rdp
from privacy_accounting.rdp_utils import convert_rdp_to_approx_dp, compose_rdp_curves
from privacy_accounting.selection_accounting import (
    compute_top1_rdp,
    compute_top_m_rdp,
    compute_two_stage_rdp,
)


PLAN_FILENAMES = {
    "papernot_baseline": {
        1: "papernot_baseline.JSON",
        2: "papernot_baseline.JSON",
    },
    "two_stage_tuning": {
        1: "two_stage_tuning_stage_1.JSON",
        2: "two_stage_tuning_stage_2.JSON",
    },
}
RESULT_FILENAMES = {
    "papernot_baseline": "papernot_baseline_results.JSON",
    "two_stage_tuning": "two_stage_tuning_results.JSON",
}
# UserAVG multiplies this seed by as much as 500 before passing it
# to NumPy's legacy uint32 RNG. Leave enough headroom for the round,
# local-step, and user offsets added during training.
MAX_TRAINING_BASE_SEED = 8_000_000


def get_simulation_method(config) -> str:
    method = str(config.simulation.method)
    if method not in PLAN_FILENAMES:
        available_methods = ", ".join(
            sorted(PLAN_FILENAMES)
        )
        raise ValueError(
            f"Unknown simulation method {method!r}. "
            f"Available methods: {available_methods}."
        )
    return method


def build_required_stage_run_specs(
    required_hp_configuration_runs,
    hp_configuration_ids,
    include_stage_2,
    plan_seed,
):
    stage_1_specs = []
    stage_2_specs = []
    hp_id_to_index = {
        str(hp_id): index
        for index, hp_id in enumerate(hp_configuration_ids)
    }

    for hp_id in hp_configuration_ids:
        hp_id = str(hp_id)
        required_runs = int(
            required_hp_configuration_runs[hp_id]
        )
        for stage_1_run_index in range(required_runs):
            stage_1_run_directory = (
                f"{hp_id}/run_{stage_1_run_index}"
            )
            stage_1_base_seed = derive_stage_1_base_seed(
                plan_seed=plan_seed,
                hp_index=hp_id_to_index[hp_id],
                stage_1_run_index=stage_1_run_index,
            )
            stage_1_specs.append(
                {
                    "hp_configuration_id": hp_id,
                    "stage_1_run_index": stage_1_run_index,
                    "stage_1_base_seed": stage_1_base_seed,
                    "stage_1_run_directory": (
                        stage_1_run_directory
                    ),
                }
            )

            if include_stage_2:
                continuation_index = 0
                stage_2_base_seed = (
                    derive_stage_2_base_seed(
                        plan_seed=plan_seed,
                        hp_index=hp_id_to_index[hp_id],
                        stage_1_run_index=stage_1_run_index,
                        continuation_index=continuation_index,
                    )
                )
                stage_2_specs.append(
                    {
                        "hp_configuration_id": hp_id,
                        "stage_1_run_index": stage_1_run_index,
                        "stage_1_base_seed": stage_1_base_seed,
                        "continuation_index": continuation_index,
                        "stage_2_base_seed": stage_2_base_seed,
                        "stage_1_run_directory": (
                            stage_1_run_directory
                        ),
                        "stage_2_run_directory": (
                            f"{stage_1_run_directory}/"
                            f"continuation_{continuation_index}"
                        ),
                    }
                )

    stage_1_base_seeds = [
        spec["stage_1_base_seed"]
        for spec in stage_1_specs
    ]
    if len(stage_1_base_seeds) != len(set(stage_1_base_seeds)):
        raise RuntimeError(
            "Derived Stage-1 base seeds collided. Change the "
            "plan seed and regenerate the plans."
        )

    stage_2_base_seeds = [
        spec["stage_2_base_seed"]
        for spec in stage_2_specs
    ]
    if len(stage_2_base_seeds) != len(set(stage_2_base_seeds)):
        raise RuntimeError(
            "Derived Stage-2 base seeds collided. Change the "
            "plan seed and regenerate the plans."
        )
    if set(stage_1_base_seeds).intersection(stage_2_base_seeds):
        raise RuntimeError(
            "Derived Stage-1 and Stage-2 base seeds collided. "
            "Change the plan seed and regenerate the plans."
        )

    return stage_1_specs, stage_2_specs


def derive_stage_1_base_seed(
    plan_seed,
    hp_index,
    stage_1_run_index,
):
    components = (
        int(plan_seed),
        1,
        int(hp_index),
        int(stage_1_run_index),
    )
    return derive_training_base_seed(components)


def derive_stage_2_base_seed(
    plan_seed,
    hp_index,
    stage_1_run_index,
    continuation_index,
):
    components = (
        int(plan_seed),
        2,
        int(hp_index),
        int(stage_1_run_index),
        int(continuation_index),
    )
    return derive_training_base_seed(components)


def derive_training_base_seed(components):
    if any(component < 0 for component in components):
        raise ValueError(
            f"Seed components must be non-negative; got {components!r}."
        )

    raw_seed = int(
        np.random.SeedSequence(components).generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )
    return raw_seed % MAX_TRAINING_BASE_SEED


def get_selected_learning_rate(config: DictConfig) -> float:
    exp_config = config.experiment
    hp_configuration_id = str(
        exp_config.simulation.run_hp_configuration
    )

    if hp_configuration_id not in config.hp_candidate_set:
        available_ids = ", ".join(
            config.hp_candidate_set.keys()
        )
        raise ValueError(
            "Unknown hyperparameter configuration "
            f"{hp_configuration_id!r}. Available configurations: "
            f"{available_ids}"
        )

    hp_configuration = config.hp_candidate_set[
        hp_configuration_id
    ]

    if "step_size" not in hp_configuration:
        raise ValueError(
            "Hyperparameter configuration "
            f"{hp_configuration_id!r} does not define "
            "'step_size'."
        )

    learning_rate = float(
        hp_configuration.step_size
    )

    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(
            f"Hyperparameter configuration {hp_configuration_id!r} "
            "must define a finite, positive 'step_size'; "
            f"got {learning_rate!r}."
        )

    return learning_rate


def generate_plan(
    config,
    method,
    m,
    E_K_values,
    num_trials,
    run_id,
    hp_configuration_ids,
    plan_filename,
):
    if method not in PLAN_FILENAMES:
        raise ValueError(
            f"Unknown plan method {method!r}."
        )

    # Total number of appearances across the complete plan.
    hp_configuration_counts = {
        hp_id: 0
        for hp_id in hp_configuration_ids
    }

    # Maximum number of appearances within any single sampled HP list.
    required_hp_configuration_runs = {
        hp_id: 0
        for hp_id in hp_configuration_ids
    }

    points = []
    total_num_simulations = 0

    for point_index, E_K in enumerate(E_K_values):
        gamma = _solve_gamma_for_conditional_mean(
            eta=config.eta,
            m=m,
            target_mean=E_K,
        )

        tnb = TNBDistribution(
            config.eta,
            gamma,
        )

        trials = []

        for trial in range(num_trials):
            rng = np.random.default_rng(
                seed=trial + config.seed + point_index
            )

            num_runs = int(
                tnb.sample_conditional(m, rng)
            )

            sampled_hp_configuration_ids = (
                rng.choice(
                    hp_configuration_ids,
                    size=num_runs,
                    replace=True,
                )
                .tolist()
            )

            trials.append(
                {
                    "trial": trial,
                    "sampled_K": num_runs,
                    "sampled_hp_configuration_ids": (
                        sampled_hp_configuration_ids
                    ),
                }
            )

            total_num_simulations += num_runs

            # Count multiplicities within this particular
            # E[K]-trial combination.
            current_hp_counts = Counter(
                sampled_hp_configuration_ids
            )

            for hp_id in hp_configuration_ids:
                current_count = current_hp_counts.get(
                    hp_id,
                    0,
                )

                # Cumulative planned appearances.
                hp_configuration_counts[
                    hp_id
                ] += current_count

                # Global maximum multiplicity.
                required_hp_configuration_runs[
                    hp_id
                ] = max(
                    required_hp_configuration_runs[hp_id],
                    current_count,
                )

        points.append(
            {
                "E_K": float(E_K),
                "gamma": float(gamma),
                "trials": trials,
            }
        )

    total_num_required_runs = sum(
        required_hp_configuration_runs.values()
    )
    include_stage_2_specs = method == "papernot_baseline"
    (
        required_stage_1_run_specs,
        required_stage_2_run_specs,
    ) = build_required_stage_run_specs(
        required_hp_configuration_runs,
        hp_configuration_ids,
        include_stage_2=include_stage_2_specs,
        plan_seed=config.seed,
    )

    plan = {
        "method": method,
        "selection_method": "papernot_top1",
        "eta": float(config.eta),
        "run_id": str(run_id),
        "plan_seed": int(config.seed),
        "hp_configuration_ids": [
            str(hp_id)
            for hp_id in hp_configuration_ids
        ],
        "points": points,
        "execution_summary": {
            # Number of HP occurrences in the complete plan.
            "total_num_simulations": total_num_simulations,
            "hp_configuration_counts": (
                hp_configuration_counts
            ),

            # Number of actual DP-FedAvg trajectories that will
            # be generated under the global-maximum reuse scheme.
            "total_num_required_runs": total_num_required_runs,
            "required_hp_configuration_runs": (
                required_hp_configuration_runs
            ),
            "required_stage_1_run_specs": (
                required_stage_1_run_specs
            ),
            "required_stage_2_run_specs": (
                required_stage_2_run_specs
            ),
        },
    }
    plan = map_stage_1_plan_runs(
        plan,
        hp_configuration_ids,
    )

    if include_stage_2_specs:
        stage_2_spec_lookup = {
            (
                spec["hp_configuration_id"],
                spec["stage_1_run_index"],
            ): spec
            for spec in required_stage_2_run_specs
        }
        for point in plan["points"]:
            for trial in point["trials"]:
                trial["sampled_stage_2_runs"] = [
                    {
                        "sample_index": run["sample_index"],
                        **stage_2_spec_lookup[
                            (
                                run["hp_configuration_id"],
                                run["stage_1_run_index"],
                            )
                        ],
                    }
                    for run in trial["sampled_stage_1_runs"]
                ]

    results_root = Path(
        config.output.results_root
    )

    experiment_name = str(
        config.name
    )

    plan_directory = (
        results_root
        / experiment_name
        / str(run_id)
        / "plan"
    )

    plan_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plan_path = (
        plan_directory
        / plan_filename
    )

    with plan_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            plan,
            file,
            indent=4,
            allow_nan=False,
        )

    return plan_path


def load_stage_1_plan(
    config,
    plan_filename="two_stage_tuning_stage_1.JSON",
):
    plan_path = (
        Path(config.output.results_root)
        / str(config.name)
        / str(config.run_id)
        / "plan"
        / plan_filename
    )

    if not plan_path.is_file():
        raise FileNotFoundError(
            "Stage-1 plan does not exist: "
            f"{plan_path}"
        )

    try:
        with plan_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            plan = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Stage-1 plan is not valid JSON: {plan_path}"
        ) from error

    if not isinstance(plan, dict):
        raise ValueError(
            "Stage-1 plan must contain a JSON object at its root: "
            f"{plan_path}"
        )

    if not isinstance(plan.get("points"), list):
        raise ValueError(
            "Stage-1 plan must contain a 'points' list: "
            f"{plan_path}"
        )

    if not isinstance(plan.get("execution_summary"), dict):
        raise ValueError(
            "Stage-1 plan must contain an "
            "'execution_summary' object: "
            f"{plan_path}"
        )

    return plan


def map_stage_1_plan_runs(
    stage_1_plan,
    hp_configuration_ids,
):
    normalized_hp_configuration_ids = [
        str(hp_id)
        for hp_id in hp_configuration_ids
    ]
    if stage_1_plan.get("hp_configuration_ids") != (
        normalized_hp_configuration_ids
    ):
        raise ValueError(
            "Stage-1 plan hp_configuration_ids do not match the "
            "experiment configuration. Regenerate the plan."
        )
    known_hp_configuration_ids = set(
        normalized_hp_configuration_ids
    )
    hp_id_to_index = {
        str(hp_id): index
        for index, hp_id in enumerate(
            normalized_hp_configuration_ids
        )
    }
    plan_seed = stage_1_plan.get("plan_seed")
    if (
        isinstance(plan_seed, bool)
        or not isinstance(plan_seed, int)
        or plan_seed < 0
    ):
        raise ValueError(
            "Stage-1 plan_seed must be a non-negative integer. "
            "Regenerate the plan with the current code."
        )

    for point_index, point in enumerate(
        stage_1_plan["points"]
    ):
        if not isinstance(point, dict):
            raise ValueError(
                "Each Stage-1 plan point must be an object; "
                f"point {point_index} is invalid."
            )

        trials = point.get("trials")
        if not isinstance(trials, list):
            raise ValueError(
                "Each Stage-1 plan point must contain a 'trials' "
                f"list; point {point_index} is invalid."
            )

        for trial_index, trial in enumerate(trials):
            if not isinstance(trial, dict):
                raise ValueError(
                    "Each Stage-1 trial must be an object; "
                    f"point {point_index}, trial {trial_index} "
                    "is invalid."
                )

            sampled_hp_ids = trial.get(
                "sampled_hp_configuration_ids"
            )
            if not isinstance(sampled_hp_ids, list):
                raise ValueError(
                    "Each Stage-1 trial must contain a "
                    "'sampled_hp_configuration_ids' list; "
                    f"point {point_index}, trial {trial_index} "
                    "is invalid."
                )

            sampled_k = trial.get("sampled_K")
            if sampled_k != len(sampled_hp_ids):
                raise ValueError(
                    "Stage-1 sampled_K does not match the number "
                    "of sampled hyperparameter configurations at "
                    f"point {point_index}, trial {trial_index}: "
                    f"sampled_K={sampled_k!r}, "
                    f"list length={len(sampled_hp_ids)}."
                )

            hp_occurrence_counts = Counter()
            sampled_stage_1_runs = []

            for sample_index, hp_id in enumerate(sampled_hp_ids):
                hp_id = str(hp_id)
                if hp_id not in known_hp_configuration_ids:
                    raise ValueError(
                        "Unknown hyperparameter configuration "
                        f"{hp_id!r} at point {point_index}, "
                        f"trial {trial_index}, sample "
                        f"{sample_index}."
                    )

                stage_1_run_index = hp_occurrence_counts[hp_id]
                hp_occurrence_counts[hp_id] += 1

                sampled_stage_1_runs.append(
                    {
                        "sample_index": sample_index,
                        "hp_configuration_id": hp_id,
                        "stage_1_run_index": stage_1_run_index,
                        "stage_1_base_seed": (
                            derive_stage_1_base_seed(
                                plan_seed=plan_seed,
                                hp_index=hp_id_to_index[hp_id],
                                stage_1_run_index=(
                                    stage_1_run_index
                                ),
                            )
                        ),
                        "stage_1_run_directory": (
                            f"{hp_id}/run_{stage_1_run_index}"
                        ),
                    }
                )

            trial["sampled_stage_1_runs"] = (
                sampled_stage_1_runs
            )

    return stage_1_plan


def normalize_metric_name(metric):
    normalized_metric = (
        str(metric).strip().lower().replace(" ", "_")
    )
    if not normalized_metric:
        raise ValueError("Metric names must not be empty.")
    return normalized_metric


def load_metric_segment(
    csv_path,
    stage,
    expected_start_round,
    expected_final_round,
    metrics,
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

    expected_rounds = list(
        range(expected_start_round, expected_final_round + 1)
    )
    observed_rounds = [row["round"] for row in rows]
    if observed_rounds != expected_rounds:
        raise ValueError(
            f"Stage-{stage} metrics CSV must contain exactly the "
            f"contiguous rounds {expected_start_round} through "
            f"{expected_final_round}: {csv_path}. Observed rounds: "
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


def load_stage_1_metric(
    csv_path,
    metric,
    evaluation_mode,
    expected_final_round,
):
    evaluation = evaluate_metric_trajectory(
        metric_segments=[
            {
                "stage": 1,
                "csv_path": csv_path,
                "expected_start_round": 0,
                "expected_final_round": expected_final_round,
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


def record_stage_1_run_scores(
    stage_1_plan,
    config,
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

    stage_1_plan["evaluation"] = {
        "metric": metric,
        "mode": evaluation_mode,
        "selection_mode": get_metric_selection_mode(
            metric=metric,
            evaluation_mode=evaluation_mode,
        ),
        "expected_final_round": expected_final_round,
    }

    simulations_root = (
        Path(config.output.results_root)
        / str(config.name)
        / str(config.run_id)
        / "simulations"
    )
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
                        )
                    )

                sampled_run.update(score_cache[run_directory])
                sampled_run["stage_1_metrics_path"] = str(
                    csv_path
                )

    return stage_1_plan


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


def build_stage_2_plan(
    stage_1_plan,
    config,
):
    if stage_1_plan.get("method") != "two_stage_tuning":
        raise ValueError(
            "A two-stage Stage-2 plan must be built from a "
            "two_stage_tuning Stage-1 plan; got "
            f"{stage_1_plan.get('method')!r}."
        )
    configured_hp_ids = [
        str(hp_id)
        for hp_id in config.hp_configuration_ids
    ]
    if stage_1_plan.get("hp_configuration_ids") != configured_hp_ids:
        raise ValueError(
            "Stage-1 plan hp_configuration_ids do not match the "
            "Stage-2 experiment configuration."
        )
    required_stage_1_specs = stage_1_plan.get(
        "execution_summary",
        {},
    ).get("required_stage_1_run_specs")
    if not isinstance(required_stage_1_specs, list):
        raise ValueError(
            "The Stage-1 plan is missing structured Stage-1 run "
            "specifications. Regenerate it with the current code."
        )

    stage_1_top_m = int(stage_1_plan["stage_1_top_m"])
    stage_2_expected_k = float(
        config.n_stage_tuning.E_K_each_stage[0]
    )
    if (
        not np.isfinite(stage_2_expected_k)
        or stage_2_expected_k < 1
    ):
        raise ValueError(
            "Stage-2 E[K] must be finite and at least 1; "
            f"got {stage_2_expected_k!r}."
        )

    stage_2_gamma = _solve_gamma_for_conditional_mean(
        eta=config.eta,
        m=1,
        target_mean=stage_2_expected_k,
    )
    stage_2_tnb = TNBDistribution(
        config.eta,
        stage_2_gamma,
    )

    points = []
    total_stage_2_appearances = 0
    stage_2_appearance_counts = Counter()
    required_stage_2_runs = Counter()
    stage_1_run_metadata = {}

    for point_index, stage_1_point in enumerate(
        stage_1_plan["points"]
    ):
        stage_2_trials = []

        for trial_index, stage_1_trial in enumerate(
            stage_1_point["trials"]
        ):
            trial_id = int(
                stage_1_trial.get("trial", trial_index)
            )
            top_m_runs = stage_1_trial[
                "top_m_stage_1_runs"
            ]
            if len(top_m_runs) != stage_1_top_m:
                raise ValueError(
                    f"Expected {stage_1_top_m} Stage-1 survivors "
                    f"at point {point_index}, trial {trial_index}; "
                    f"got {len(top_m_runs)}."
                )

            survivor_directories = [
                run["stage_1_run_directory"]
                for run in top_m_runs
            ]
            if len(set(survivor_directories)) != stage_1_top_m:
                raise ValueError(
                    "Stage-1 survivors must refer to distinct "
                    "independent run directories at point "
                    f"{point_index}, trial {trial_index}."
                )

            for run in top_m_runs:
                run_directory = run["stage_1_run_directory"]
                run_metadata = {
                    "hp_configuration_id": str(
                        run["hp_configuration_id"]
                    ),
                    "stage_1_run_index": int(
                        run["stage_1_run_index"]
                    ),
                    "stage_1_base_seed": int(
                        run["stage_1_base_seed"]
                    ),
                }
                previous_metadata = stage_1_run_metadata.get(
                    run_directory
                )
                if (
                    previous_metadata is not None
                    and previous_metadata != run_metadata
                ):
                    raise ValueError(
                        "Inconsistent metadata for Stage-1 run "
                        f"{run_directory!r}."
                    )
                stage_1_run_metadata[run_directory] = run_metadata

            sampling_seed = [
                int(config.seed),
                2,
                point_index,
                trial_id,
            ]
            rng = np.random.default_rng(
                np.random.SeedSequence(sampling_seed)
            )
            sampled_k = int(
                stage_2_tnb.sample_conditional(1, rng)
            )
            sampled_survivor_directories = (
                rng.choice(
                    survivor_directories,
                    size=sampled_k,
                    replace=True,
                )
                .tolist()
            )

            current_appearance_counts = Counter(
                sampled_survivor_directories
            )
            total_stage_2_appearances += sampled_k
            stage_2_appearance_counts.update(
                current_appearance_counts
            )
            for run_directory, count in (
                current_appearance_counts.items()
            ):
                required_stage_2_runs[run_directory] = max(
                    required_stage_2_runs[run_directory],
                    count,
                )

            continuation_counts = Counter()
            sampled_stage_2_runs = []
            for sample_index, run_directory in enumerate(
                sampled_survivor_directories
            ):
                continuation_index = continuation_counts[
                    run_directory
                ]
                continuation_counts[run_directory] += 1

                sampled_stage_2_runs.append(
                    {
                        "sample_index": sample_index,
                        "stage_1_run_directory": run_directory,
                        "continuation_index": continuation_index,
                        "stage_2_run_directory": (
                            f"{run_directory}/"
                            f"continuation_{continuation_index}"
                        ),
                    }
                )

            stage_2_trials.append(
                {
                    "trial": trial_id,
                    "trial_stage_1": stage_1_trial,
                    "trial_stage_2": {
                        "sampling_seed": sampling_seed,
                        "candidate_stage_1_run_directories": (
                            survivor_directories
                        ),
                        "sampled_K": sampled_k,
                        "sampled_hp_configuration_ids": (
                            sampled_survivor_directories
                        ),
                        "sampled_stage_2_runs": (
                            sampled_stage_2_runs
                        ),
                    },
                }
            )

        points.append(
            {
                "stage_1_E_K": float(stage_1_point["E_K"]),
                "stage_1_gamma": float(
                    stage_1_point["gamma"]
                ),
                "stage_2_E_K": stage_2_expected_k,
                "stage_2_gamma": float(stage_2_gamma),
                "trials": stage_2_trials,
            }
        )

    required_run_directories = []
    required_run_specs = []
    hp_id_to_index = {
        str(hp_id): index
        for index, hp_id in enumerate(configured_hp_ids)
    }
    for run_directory, count in sorted(
        required_stage_2_runs.items()
    ):
        run_metadata = stage_1_run_metadata[run_directory]
        for continuation_index in range(count):
            stage_2_run_directory = (
                f"{run_directory}/"
                f"continuation_{continuation_index}"
            )
            required_run_directories.append(
                stage_2_run_directory
            )
            stage_2_base_seed = derive_stage_2_base_seed(
                plan_seed=config.seed,
                hp_index=hp_id_to_index[
                    run_metadata["hp_configuration_id"]
                ],
                stage_1_run_index=(
                    run_metadata["stage_1_run_index"]
                ),
                continuation_index=continuation_index,
            )
            required_run_specs.append(
                {
                    **run_metadata,
                    "continuation_index": continuation_index,
                    "stage_2_base_seed": stage_2_base_seed,
                    "stage_1_run_directory": run_directory,
                    "stage_2_run_directory": (
                        stage_2_run_directory
                    ),
                }
            )

    stage_2_base_seeds = [
        spec["stage_2_base_seed"]
        for spec in required_run_specs
    ]
    if len(stage_2_base_seeds) != len(set(stage_2_base_seeds)):
        raise RuntimeError(
            "Derived Stage-2 base seeds collided. Change the "
            "plan seed and regenerate the plans."
        )
    stage_1_base_seeds = {
        int(spec["stage_1_base_seed"])
        for spec in required_stage_1_specs
    }
    if stage_1_base_seeds.intersection(stage_2_base_seeds):
        raise RuntimeError(
            "Derived Stage-1 and Stage-2 base seeds collided. "
            "Change the plan seed and regenerate the plans."
        )

    return {
        "method": "two_stage_tuning",
        "selection_method": "papernot_top1",
        "eta": float(config.eta),
        "run_id": str(config.run_id),
        "plan_seed": int(config.seed),
        "hp_configuration_ids": list(configured_hp_ids),
        "stage_1_top_m": stage_1_top_m,
        "stage_2_selection_m": 1,
        "evaluation": dict(stage_1_plan["evaluation"]),
        "points": points,
        "execution_summary": {
            "required_stage_1_run_specs": list(
                required_stage_1_specs
            ),
            "total_num_stage_2_appearances": (
                total_stage_2_appearances
            ),
            "stage_2_appearance_counts": dict(
                sorted(stage_2_appearance_counts.items())
            ),
            "total_num_required_stage_2_runs": sum(
                required_stage_2_runs.values()
            ),
            "required_stage_2_runs": dict(
                sorted(required_stage_2_runs.items())
            ),
            "required_stage_2_run_directories": (
                required_run_directories
            ),
            "required_stage_2_run_specs": required_run_specs,
        },
    }


def generate_stage_2_plan(
    stage_1_plan,
    config,
    plan_filename="two_stage_tuning_stage_2.JSON",
):
    stage_2_plan = build_stage_2_plan(
        stage_1_plan,
        config,
    )
    plan_directory = (
        Path(config.output.results_root)
        / str(config.name)
        / str(config.run_id)
        / "plan"
    )
    plan_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    plan_path = plan_directory / plan_filename

    with plan_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        encoder = json.JSONEncoder(
            indent=4,
            allow_nan=False,
        )
        pending_characters = 0

        for chunk in encoder.iterencode(stage_2_plan):
            file.write(chunk)
            pending_characters += len(chunk)

            if pending_characters >= 1_000_000:
                file.flush()
                pending_characters = 0

    return plan_path


def load_stage_2_plan(
    config,
    plan_filename="two_stage_tuning_stage_2.JSON",
):
    plan_path = (
        Path(config.output.results_root)
        / str(config.name)
        / str(config.run_id)
        / "plan"
        / plan_filename
    )
    if not plan_path.is_file():
        raise FileNotFoundError(
            "Stage-2 plan does not exist: "
            f"{plan_path}"
        )

    try:
        with plan_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            plan = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Stage-2 plan is not valid JSON: {plan_path}"
        ) from error

    if not isinstance(plan, dict):
        raise ValueError(
            "Stage-2 plan must contain a JSON object at its root: "
            f"{plan_path}"
        )
    execution_summary = plan.get("execution_summary")
    if not isinstance(execution_summary, dict):
        raise ValueError(
            "Stage-2 plan must contain an "
            f"'execution_summary' object: {plan_path}"
        )
    if not isinstance(
        execution_summary.get("required_stage_2_run_specs"),
        list,
    ):
        raise ValueError(
            "Stage-2 plan must contain a "
            "'required_stage_2_run_specs' list. Regenerate the "
            f"plan with the current code: {plan_path}"
        )

    return plan


def get_required_stage_2_run_specs(
    stage_2_plan,
    hp_configuration_id,
):
    hp_configuration_id = str(hp_configuration_id)
    plan_hp_ids = stage_2_plan.get("hp_configuration_ids")
    if (
        not isinstance(plan_hp_ids, list)
        or not plan_hp_ids
        or len(plan_hp_ids) != len(set(plan_hp_ids))
    ):
        raise ValueError(
            "The Stage-2 plan must contain a non-empty, unique "
            "hp_configuration_ids list. Regenerate the plan."
        )
    hp_id_to_index = {
        str(hp_id): index
        for index, hp_id in enumerate(plan_hp_ids)
    }
    plan_seed = stage_2_plan.get("plan_seed")
    if (
        isinstance(plan_seed, bool)
        or not isinstance(plan_seed, int)
        or plan_seed < 0
    ):
        raise ValueError(
            "Stage-2 plan_seed must be a non-negative integer."
        )

    required_keys = {
        "hp_configuration_id",
        "stage_1_run_index",
        "stage_1_base_seed",
        "continuation_index",
        "stage_2_base_seed",
        "stage_1_run_directory",
        "stage_2_run_directory",
    }
    selected_specs = []
    observed_stage_2_base_seeds = set()

    for spec_index, spec in enumerate(
        stage_2_plan["execution_summary"][
            "required_stage_2_run_specs"
        ]
    ):
        if not isinstance(spec, dict):
            raise ValueError(
                "Each required Stage-2 run specification must "
                f"be an object; entry {spec_index} is invalid."
            )
        missing_keys = required_keys.difference(spec)
        if missing_keys:
            raise ValueError(
                f"Required Stage-2 run entry {spec_index} is "
                f"missing fields: {sorted(missing_keys)}."
            )

        spec_hp_id = str(spec["hp_configuration_id"])
        stage_1_run_index = spec["stage_1_run_index"]
        stage_1_base_seed = spec["stage_1_base_seed"]
        continuation_index = spec["continuation_index"]
        stage_2_base_seed = spec["stage_2_base_seed"]
        if spec_hp_id not in hp_id_to_index:
            raise ValueError(
                f"Required Stage-2 run entry {spec_index} contains "
                f"unknown HP configuration {spec_hp_id!r}."
            )
        if (
            isinstance(stage_1_run_index, bool)
            or not isinstance(stage_1_run_index, int)
            or stage_1_run_index < 0
            or isinstance(stage_1_base_seed, bool)
            or not isinstance(stage_1_base_seed, int)
            or stage_1_base_seed < 0
            or stage_1_base_seed >= MAX_TRAINING_BASE_SEED
            or isinstance(continuation_index, bool)
            or not isinstance(continuation_index, int)
            or continuation_index < 0
            or isinstance(stage_2_base_seed, bool)
            or not isinstance(stage_2_base_seed, int)
            or stage_2_base_seed < 0
            or stage_2_base_seed >= MAX_TRAINING_BASE_SEED
        ):
            raise ValueError(
                f"Required Stage-2 run entry {spec_index} has "
                "invalid Stage-1 or Stage-2 seed values."
            )
        if stage_2_base_seed in observed_stage_2_base_seeds:
            raise ValueError(
                "Stage-2 base seeds must be unique; "
                f"{stage_2_base_seed} is repeated."
            )
        observed_stage_2_base_seeds.add(stage_2_base_seed)
        expected_stage_1_base_seed = derive_stage_1_base_seed(
            plan_seed=plan_seed,
            hp_index=hp_id_to_index[spec_hp_id],
            stage_1_run_index=stage_1_run_index,
        )
        expected_stage_2_base_seed = derive_stage_2_base_seed(
            plan_seed=plan_seed,
            hp_index=hp_id_to_index[spec_hp_id],
            stage_1_run_index=stage_1_run_index,
            continuation_index=continuation_index,
        )
        if stage_1_base_seed != expected_stage_1_base_seed:
            raise ValueError(
                f"Required Stage-2 run entry {spec_index} has an "
                "inconsistent Stage-1 base seed."
            )
        if stage_2_base_seed != expected_stage_2_base_seed:
            raise ValueError(
                f"Required Stage-2 run entry {spec_index} has an "
                "inconsistent Stage-2 base seed."
            )

        stage_1_directory = Path(
            str(spec["stage_1_run_directory"])
        )
        stage_2_directory = Path(
            str(spec["stage_2_run_directory"])
        )
        if (
            stage_1_directory.is_absolute()
            or stage_2_directory.is_absolute()
            or ".." in stage_1_directory.parts
            or ".." in stage_2_directory.parts
        ):
            raise ValueError(
                f"Required Stage-2 run entry {spec_index} contains "
                "an unsafe absolute or parent-relative path."
            )
        expected_stage_1_directory = Path(
            spec_hp_id,
            f"run_{stage_1_run_index}",
        )
        expected_stage_2_directory = (
            expected_stage_1_directory
            / f"continuation_{continuation_index}"
        )
        if stage_1_directory != expected_stage_1_directory:
            raise ValueError(
                f"Required Stage-2 run entry {spec_index} has "
                "inconsistent Stage-1 metadata."
            )
        if stage_2_directory != expected_stage_2_directory:
            raise ValueError(
                f"Required Stage-2 run entry {spec_index} has "
                "inconsistent Stage-2 metadata."
            )

        if spec_hp_id == hp_configuration_id:
            selected_specs.append(dict(spec))

    if not selected_specs:
        raise ValueError(
            "The Stage-2 plan has no required runs for "
            f"hyperparameter configuration {hp_configuration_id!r}."
        )

    return selected_specs


def get_required_stage_1_run_specs(
    stage_1_plan,
    hp_configuration_id,
):
    hp_configuration_id = str(hp_configuration_id)
    plan_hp_ids = stage_1_plan.get("hp_configuration_ids")
    if (
        not isinstance(plan_hp_ids, list)
        or not plan_hp_ids
        or len(plan_hp_ids) != len(set(plan_hp_ids))
    ):
        raise ValueError(
            "The Stage-1 plan must contain a non-empty, unique "
            "hp_configuration_ids list. Regenerate the plan."
        )
    hp_id_to_index = {
        str(hp_id): index
        for index, hp_id in enumerate(plan_hp_ids)
    }
    plan_seed = stage_1_plan.get("plan_seed")
    if (
        isinstance(plan_seed, bool)
        or not isinstance(plan_seed, int)
        or plan_seed < 0
    ):
        raise ValueError(
            "Stage-1 plan_seed must be a non-negative integer."
        )

    required_keys = {
        "hp_configuration_id",
        "stage_1_run_index",
        "stage_1_base_seed",
        "stage_1_run_directory",
    }
    selected_specs = []
    observed_stage_1_base_seeds = set()

    for spec_index, spec in enumerate(
        stage_1_plan["execution_summary"][
            "required_stage_1_run_specs"
        ]
    ):
        if not isinstance(spec, dict):
            raise ValueError(
                "Each required Stage-1 run specification must "
                f"be an object; entry {spec_index} is invalid."
            )
        missing_keys = required_keys.difference(spec)
        if missing_keys:
            raise ValueError(
                f"Required Stage-1 run entry {spec_index} is "
                f"missing fields: {sorted(missing_keys)}."
            )

        spec_hp_id = str(spec["hp_configuration_id"])
        stage_1_run_index = spec["stage_1_run_index"]
        stage_1_base_seed = spec["stage_1_base_seed"]
        if spec_hp_id not in hp_id_to_index:
            raise ValueError(
                f"Required Stage-1 run entry {spec_index} contains "
                f"unknown HP configuration {spec_hp_id!r}."
            )
        if (
            isinstance(stage_1_run_index, bool)
            or not isinstance(stage_1_run_index, int)
            or stage_1_run_index < 0
            or isinstance(stage_1_base_seed, bool)
            or not isinstance(stage_1_base_seed, int)
            or stage_1_base_seed < 0
            or stage_1_base_seed >= MAX_TRAINING_BASE_SEED
        ):
            raise ValueError(
                f"Required Stage-1 run entry {spec_index} has "
                "invalid seed values."
            )
        if stage_1_base_seed in observed_stage_1_base_seeds:
            raise ValueError(
                "Stage-1 base seeds must be unique; "
                f"{stage_1_base_seed} is repeated."
            )
        observed_stage_1_base_seeds.add(stage_1_base_seed)
        expected_stage_1_base_seed = derive_stage_1_base_seed(
            plan_seed=plan_seed,
            hp_index=hp_id_to_index[spec_hp_id],
            stage_1_run_index=stage_1_run_index,
        )
        if stage_1_base_seed != expected_stage_1_base_seed:
            raise ValueError(
                f"Required Stage-1 run entry {spec_index} has an "
                "inconsistent Stage-1 base seed."
            )

        stage_1_directory = Path(
            str(spec["stage_1_run_directory"])
        )
        expected_directory = Path(
            spec_hp_id,
            f"run_{stage_1_run_index}",
        )
        if (
            stage_1_directory.is_absolute()
            or ".." in stage_1_directory.parts
            or stage_1_directory != expected_directory
        ):
            raise ValueError(
                f"Required Stage-1 run entry {spec_index} has "
                "inconsistent or unsafe path metadata."
            )

        if spec_hp_id == hp_configuration_id:
            selected_specs.append(dict(spec))

    if not selected_specs:
        raise ValueError(
            "The simulation plan has no required Stage-1 runs "
            f"for hyperparameter configuration "
            f"{hp_configuration_id!r}."
        )

    return selected_specs


def load_simulation_plan(
    config,
    stage,
):
    if stage not in {1, 2}:
        raise ValueError(
            f"Simulation stage must be 1 or 2; got {stage!r}."
        )

    method = get_simulation_method(config)
    plan_filename = PLAN_FILENAMES[method][stage]
    if stage == 1:
        plan = load_stage_1_plan(
            config,
            plan_filename=plan_filename,
        )
    else:
        plan = load_stage_2_plan(
            config,
            plan_filename=plan_filename,
        )

    if plan.get("method") != method:
        raise ValueError(
            f"Plan method {plan.get('method')!r} does not match "
            f"simulation method {method!r}."
        )

    required_specs_key = (
        f"required_stage_{stage}_run_specs"
    )
    required_specs = plan["execution_summary"].get(
        required_specs_key
    )
    if not isinstance(required_specs, list):
        raise ValueError(
            f"Simulation plan {plan_filename} does not contain "
            f"a {required_specs_key!r} list. Regenerate the plan."
        )

    return plan


def get_required_simulation_run_specs(
    plan,
    stage,
    hp_configuration_id,
):
    if stage == 1:
        return get_required_stage_1_run_specs(
            plan,
            hp_configuration_id,
        )
    if stage == 2:
        return get_required_stage_2_run_specs(
            plan,
            hp_configuration_id,
        )
    raise ValueError(
        f"Simulation stage must be 1 or 2; got {stage!r}."
    )


def validate_simulation_stage(config, stage):
    exp_config = config.experiment
    configured_stage = int(exp_config.simulation.stage)
    if configured_stage != stage:
        raise ValueError(
            f"Stage-{stage} simulation requires "
            f"simulation.stage={stage}; got "
            f"{exp_config.simulation.stage!r}."
        )

    rounds = int(config.run_settings.rounds)
    stage_1_end = int(exp_config.simulation.stage_1_end)
    if stage == 1 and rounds != stage_1_end:
        raise ValueError(
            "Stage-1 simulation requires run_settings.rounds to "
            "equal simulation.stage_1_end; got "
            f"{rounds!r} and {stage_1_end!r}."
        )
    if stage == 2 and rounds <= stage_1_end:
        raise ValueError(
            "Stage-2 simulation requires run_settings.rounds to "
            "be greater than simulation.stage_1_end; got "
            f"{rounds!r} and {stage_1_end!r}."
        )


def build_model(config):
    model_class = getattr(
        models,
        config.dataset.model_name,
    )
    if config.dataset.name == "synthetic":
        return model_class(
            input_dim=config.dataset.dim_input,
            output_dim=config.dataset.dim_output,
        )
    return model_class()


def get_learning_steps(config):
    learning_rate = get_selected_learning_rate(config)
    step_mode = str(config.server.constant_global_step)
    if step_mode == "Fixed":
        return learning_rate, config.server.global_step
    if step_mode == "Adaptive":
        global_step = (
            config.server.client_ratio
            * config.dataset.nb_users
        ) ** 0.5
        local_step = learning_rate / (
            config.server.local_updates * global_step
        )
        return local_step, global_step
    raise ValueError(
        "server.constant_global_step must be 'Fixed' or "
        f"'Adaptive'; got {step_mode!r}."
    )


def save_run_spec(save_path, stage, run_spec):
    spec_path = save_path / f"stage_{stage}_run_spec.JSON"
    if spec_path.is_file():
        try:
            with spec_path.open(
                mode="r",
                encoding="utf-8",
            ) as file:
                existing_spec = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Existing run specification is invalid: {spec_path}"
            ) from error
        if existing_spec != run_spec:
            raise ValueError(
                "The existing run specification does not match "
                f"the selected plan: {spec_path}. Use a new run_id "
                "or restore the plan that created this run."
            )

    with spec_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_spec,
            file,
            indent=4,
            allow_nan=False,
        )


def run_planned_simulations(config, stage):
    validate_simulation_stage(config, stage)
    exp_config = config.experiment
    simulation_plan = load_simulation_plan(
        exp_config,
        stage=stage,
    )
    required_run_specs = get_required_simulation_run_specs(
        simulation_plan,
        stage=stage,
        hp_configuration_id=(
            exp_config.simulation.run_hp_configuration
        ),
    )
    local_step, global_step = get_learning_steps(config)
    simulations_root = (
        Path(HydraConfig.get().runtime.output_dir)
        / "simulations"
    )

    for run_spec in required_run_specs:
        base_seed = int(
            run_spec[f"stage_{stage}_base_seed"]
        )
        set_seed(base_seed)
        save_path = (
            simulations_root
            / run_spec[f"stage_{stage}_run_directory"]
        )
        save_path.mkdir(parents=True, exist_ok=True)
        save_run_spec(
            save_path=save_path,
            stage=stage,
            run_spec=run_spec,
        )

        stage_1_source_path = None
        if stage == 2:
            stage_1_source_path = (
                simulations_root
                / run_spec["stage_1_run_directory"]
            )

        train_data_loader, test_data_loader = get_data_loaders(
            config,
            per_client_loader=True,
        )
        server = FedAvg(
            model=build_model(config),
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
            save_path=save_path,
            file_name=None,
            num_glob_iters=config.run_settings.rounds,
            loss_fn_name=config.dataset.loss_fn,
            local_learning_rate=local_step,
            global_learning_rate=global_step,
            weight_decay=config.server.weight_decay,
            use_cuda=config.run_settings.use_cuda,
            similarity=config.dataset.similarity,
            client_ratio=config.server.client_ratio,
            dp=config.server.dp,
            local_updates=config.server.local_updates,
            sample_rate=config.server.sampling_rate,
            noise_multiplier=config.server.sigma,
            max_grad_norm=config.server.max_grad_norm,
            x_label=config.dataset.x_label,
            y_label=config.dataset.y_label,
            client_sampling_scheme=(
                config.server.client_sampling_scheme
            ),
            data_sampling_scheme=(
                config.server.data_sampling_scheme
            ),
            stage=stage,
            stage_1_end=exp_config.simulation.stage_1_end,
            base_seed=base_seed,
            stage_1_source_path=stage_1_source_path,
        )
        server.train()
        OmegaConf.save(
            config,
            save_path / f"stage_{stage}_config.yaml",
            resolve=True,
        )


def get_stage_compute_schedule(config):
    stage_1_end = int(config.simulation.stage_1_end)
    stage_2_end = int(config.simulation.stage_2_end)
    if stage_1_end <= 0:
        raise ValueError(
            "simulation.stage_1_end must be positive."
        )
    if stage_2_end <= stage_1_end:
        raise ValueError(
            "simulation.stage_2_end must be greater than "
            "simulation.stage_1_end."
        )

    local_updates_schedule = [
        int(local_updates)
        for local_updates in config.local_updates_schedule
    ]
    if (
        len(local_updates_schedule) != 2
        or any(
            local_updates <= 0
            for local_updates in local_updates_schedule
        )
    ):
        raise ValueError(
            "local_updates_schedule must contain two positive "
            "integers, one for each stage."
        )

    stage_rounds = [
        stage_1_end,
        stage_2_end - stage_1_end,
    ]
    return [
        rounds * local_updates
        for rounds, local_updates in zip(
            stage_rounds,
            local_updates_schedule,
        )
    ]


def get_compilation_paths(config):
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
    return plan


def score_complete_run(
    run,
    simulations_root,
    evaluation,
    stage_1_end,
    stage_2_end,
    score_cache,
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
                },
                {
                    "stage": 2,
                    "csv_path": stage_2_metrics_path,
                    "expected_start_round": stage_1_end,
                    "expected_final_round": stage_2_end - 1,
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


def get_compiled_evaluation_metadata(
    evaluation,
    stage_1_end,
    stage_2_end,
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
            "stage_1_rounds": [0, stage_1_end - 1],
            "stage_2_rounds": [
                stage_1_end,
                stage_2_end - 1,
            ],
        },
    }


def compile_papernot_results(
    config,
    plan,
    evaluation,
    stage_compute_schedule,
):
    result = copy.deepcopy(plan)
    result["result_type"] = "compiled_hpo_results"
    result["evaluation"] = get_compiled_evaluation_metadata(
        evaluation=evaluation,
        stage_1_end=int(config.simulation.stage_1_end),
        stage_2_end=int(config.simulation.stage_2_end),
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


def compile_two_stage_results(
    config,
    plan,
    evaluation,
    stage_compute_schedule,
):
    result = copy.deepcopy(plan)
    result["result_type"] = "compiled_hpo_results"
    result["evaluation"] = get_compiled_evaluation_metadata(
        evaluation=evaluation,
        stage_1_end=int(config.simulation.stage_1_end),
        stage_2_end=int(config.simulation.stage_2_end),
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


def save_trial_result_rows(rows, csv_path):
    if not rows:
        raise ValueError("No trial result rows were compiled.")
    with csv_path.open(
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


def plot_expected_compute_utility(rows, output_directory):
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

        axis.set_xlabel(
            "Expected compute (communication rounds × local updates)"
        )
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


def compile_experiment_results(config):
    evaluation = get_evaluation_settings(config)
    stage_compute_schedule = get_stage_compute_schedule(config)
    paths = get_compilation_paths(config)
    plans = {
        method: load_compilation_plan(config, method)
        for method in RESULT_FILENAMES
    }
    compiled_results = {
        "papernot_baseline": compile_papernot_results(
            config=config,
            plan=plans["papernot_baseline"],
            evaluation=evaluation,
            stage_compute_schedule=stage_compute_schedule,
        ),
        "two_stage_tuning": compile_two_stage_results(
            config=config,
            plan=plans["two_stage_tuning"],
            evaluation=evaluation,
            stage_compute_schedule=stage_compute_schedule,
        ),
    }

    compiled_root = paths["compiled_root"]
    compiled_root.mkdir(parents=True, exist_ok=True)
    result_paths = {}
    for method, result in compiled_results.items():
        plan_path = (
            paths["plan_root"]
            / PLAN_FILENAMES[method][2]
        )
        result["source_plan_path"] = str(plan_path)
        result_path = compiled_root / RESULT_FILENAMES[method]
        with result_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            encoder = json.JSONEncoder(
                indent=4,
                allow_nan=False,
            )
            pending_characters = 0

            for chunk in encoder.iterencode(result):
                file.write(chunk)
                pending_characters += len(chunk)

                if pending_characters >= 1_000_000:
                    file.flush()
                    pending_characters = 0
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
    )

    return {
        "result_paths": result_paths,
        "trial_csv_path": trial_csv_path,
        "plot_paths": plot_paths,
    }

def calculate_E_K_given_compute_for_papernot(compute, local_updates_schedule):
    E_K = compute/sum(local_updates_schedule)
    return E_K


def calculate_compute_given_E_K_two_stage_tuning(config, E_K, local_updates_schedule):
    E_K_each_stage = [E_K] + config.n_stage_tuning.E_K_each_stage
    compute = 0
    for i, E_K in enumerate(E_K_each_stage):
        compute += E_K * local_updates_schedule[i]
    return compute


def load_privacy_compute_points(config):
    paths = get_compilation_paths(config)
    stage_compute_schedule = get_stage_compute_schedule(config)
    compiled_results = {}

    for method, filename in RESULT_FILENAMES.items():
        result_path = paths["compiled_root"] / filename
        if not result_path.is_file():
            raise FileNotFoundError(
                "Compiled results are required before privacy "
                f"accounting: {result_path}"
            )
        try:
            with result_path.open(
                mode="r",
                encoding="utf-8",
            ) as file:
                result = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Compiled results are not valid JSON: {result_path}"
            ) from error
        if result.get("method") != method:
            raise ValueError(
                f"Compiled result {result_path} has method "
                f"{result.get('method')!r}, expected {method!r}."
            )
        compiled_results[method] = result

    papernot_result = compiled_results["papernot_baseline"]
    two_stage_result = compiled_results["two_stage_tuning"]
    papernot_points = []
    for point_index, point in enumerate(
        papernot_result["points"]
    ):
        expected_num_trials = float(point["E_K"])
        expected_compute = float(point["expected_compute"])
        calculated_compute = (
            expected_num_trials * sum(stage_compute_schedule)
        )
        if not np.isclose(
            expected_compute,
            calculated_compute,
            rtol=1e-12,
            atol=1e-9,
        ):
            raise ValueError(
                "Papernot compiled expected compute is inconsistent "
                f"at point {point_index}: stored={expected_compute}, "
                f"calculated={calculated_compute}."
            )
        papernot_points.append(
            {
                "point_index": point_index,
                "expected_compute": expected_compute,
                "expected_num_trials": expected_num_trials,
            }
        )

    two_stage_points = []
    for point_index, point in enumerate(
        two_stage_result["points"]
    ):
        stage_1_expected_num_trials = float(
            point["stage_1_E_K"]
        )
        stage_2_expected_num_trials = float(
            point["stage_2_E_K"]
        )
        expected_compute = float(point["expected_compute"])
        calculated_compute = (
            stage_1_expected_num_trials
            * stage_compute_schedule[0]
            + stage_2_expected_num_trials
            * stage_compute_schedule[1]
        )
        if not np.isclose(
            expected_compute,
            calculated_compute,
            rtol=1e-12,
            atol=1e-9,
        ):
            raise ValueError(
                "Two-stage compiled expected compute is "
                f"inconsistent at point {point_index}: "
                f"stored={expected_compute}, "
                f"calculated={calculated_compute}."
            )
        two_stage_points.append(
            {
                "point_index": point_index,
                "expected_compute": expected_compute,
                "stage_1_expected_num_trials": (
                    stage_1_expected_num_trials
                ),
                "stage_2_expected_num_trials": (
                    stage_2_expected_num_trials
                ),
            }
        )

    papernot_points.sort(
        key=lambda point: point["expected_compute"]
    )
    two_stage_points.sort(
        key=lambda point: point["expected_compute"]
    )
    papernot_compute = np.asarray(
        [point["expected_compute"] for point in papernot_points]
    )
    two_stage_compute = np.asarray(
        [point["expected_compute"] for point in two_stage_points]
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
            "Papernot and two-stage compiled results do not use "
            "matching expected-compute coordinates."
        )

    return {
        "papernot_points": papernot_points,
        "two_stage_points": two_stage_points,
        "papernot_eta": float(papernot_result["eta"]),
        "two_stage_eta": float(two_stage_result["eta"]),
        "two_stage_top_m": int(
            two_stage_result["stage_1_top_m"]
        ),
    }


def validate_privacy_order_search(dp_result, method, expected_compute):
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
    if not rows:
        raise ValueError("No privacy-compute result rows were produced.")
    with csv_path.open(
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


def plot_privacy_compute_plot(config):
    exp_config = config.experiment
    privacy_config = exp_config.privacy
    max_renyi_order = int(privacy_config.max_renyi_order)
    if max_renyi_order < 3:
        raise ValueError(
            "privacy.max_renyi_order must be at least 3."
        )
    orders = np.arange(2, max_renyi_order + 1)
    delta = float(privacy_config.delta)
    if not 0.0 < delta < 1.0:
        raise ValueError(
            f"privacy.delta must satisfy 0 < delta < 1; got {delta}."
        )
    accounting_method = str(
        privacy_config.accounting_method
    ).strip().lower()
    if accounting_method not in {"bounds", "numerical"}:
        raise ValueError(
            "privacy.accounting_method must be 'bounds' or "
            f"'numerical'; got {accounting_method!r}."
        )

    low_resource_config = {
        "num_rounds": int(exp_config.simulation.stage_1_end),
        "num_local_updates": int(
            exp_config.local_updates_schedule[0]
        ),
        "num_clients": int(config.dataset.nb_users),
        "client_sampling_rate": float(config.server.client_ratio),
        "local_sampling_rate": float(config.server.sampling_rate),
        "sigma_gaussian": float(config.server.sigma),
        "sigma_is_actual": False,
    }
    high_resource_config = {
        "num_rounds": int(
            exp_config.simulation.stage_2_end
            - exp_config.simulation.stage_1_end
        ),
        "num_local_updates": int(
            exp_config.local_updates_schedule[1]
        ),
        "num_clients": int(config.dataset.nb_users),
        "client_sampling_rate": float(config.server.client_ratio),
        "local_sampling_rate": float(config.server.sampling_rate),
        "sigma_gaussian": float(config.server.sigma),
        "sigma_is_actual": False,
    }
    low_resource_curve = compute_dpfedavg_rdp(
        config=low_resource_config,
        orders=orders,
        accounting_method=accounting_method,
    )
    high_resource_curve = compute_dpfedavg_rdp(
        config=high_resource_config,
        orders=orders,
        accounting_method=accounting_method,
    )
    papernot_base_curve = compose_rdp_curves(
        low_resource_curve,
        high_resource_curve,
    )
    accounting_metadata = {
        "stage_1_num_rounds": low_resource_config[
            "num_rounds"
        ],
        "stage_2_num_rounds": high_resource_config[
            "num_rounds"
        ],
        "stage_1_num_local_updates": low_resource_config[
            "num_local_updates"
        ],
        "stage_2_num_local_updates": high_resource_config[
            "num_local_updates"
        ],
        "num_clients": low_resource_config["num_clients"],
        "client_sampling_rate": low_resource_config[
            "client_sampling_rate"
        ],
        "local_sampling_rate": low_resource_config[
            "local_sampling_rate"
        ],
        "sigma_gaussian": low_resource_config[
            "sigma_gaussian"
        ],
        "sigma_is_actual": low_resource_config[
            "sigma_is_actual"
        ],
        "effective_gaussian_noise_multiplier": (
            low_resource_config["sigma_gaussian"]
            * np.sqrt(
                low_resource_config["client_sampling_rate"]
                * low_resource_config["num_clients"]
            )
        ),
        "client_sampling_scheme": str(
            config.server.client_sampling_scheme
        ),
        "data_sampling_scheme": str(
            config.server.data_sampling_scheme
        ),
    }

    privacy_points = load_privacy_compute_points(exp_config)
    privacy_rows = []
    for point in privacy_points["papernot_points"]:
        selection_result = compute_top1_rdp(
            base_rdp_curve=papernot_base_curve,
            expected_num_trials=point["expected_num_trials"],
            eta=privacy_points["papernot_eta"],
        )
        dp_result = convert_rdp_to_approx_dp(
            selection_result.rdp_curve,
            delta=delta,
        )
        validate_privacy_order_search(
            dp_result=dp_result,
            method="papernot_baseline",
            expected_compute=point["expected_compute"],
        )
        privacy_rows.append(
            {
                "method": "papernot_baseline",
                "point_index": point["point_index"],
                "expected_compute": point["expected_compute"],
                "expected_num_trials": point[
                    "expected_num_trials"
                ],
                "stage_1_expected_num_trials": "",
                "stage_2_expected_num_trials": "",
                "top_m": 1,
                "eta": privacy_points["papernot_eta"],
                "epsilon": dp_result.epsilon,
                "delta": dp_result.delta,
                "best_renyi_order": dp_result.best_order,
                "is_at_min_order": dp_result.is_at_min_order,
                "is_at_max_order": dp_result.is_at_max_order,
                "min_renyi_order": int(orders[0]),
                "max_renyi_order": int(orders[-1]),
                "accounting_method": accounting_method,
                **accounting_metadata,
            }
        )

    for point in privacy_points["two_stage_points"]:
        selection_result = compute_two_stage_rdp(
            stage_1_base_rdp_curve=low_resource_curve,
            stage_2_base_rdp_curve=high_resource_curve,
            m=privacy_points["two_stage_top_m"],
            expected_num_trials_stage_1=point[
                "stage_1_expected_num_trials"
            ],
            expected_num_trials_stage_2=point[
                "stage_2_expected_num_trials"
            ],
            eta_stage_1=privacy_points["two_stage_eta"],
            eta_stage_2=privacy_points["two_stage_eta"],
        )
        dp_result = convert_rdp_to_approx_dp(
            selection_result.rdp_curve,
            delta=delta,
        )
        validate_privacy_order_search(
            dp_result=dp_result,
            method="two_stage_tuning",
            expected_compute=point["expected_compute"],
        )
        privacy_rows.append(
            {
                "method": "two_stage_tuning",
                "point_index": point["point_index"],
                "expected_compute": point["expected_compute"],
                "expected_num_trials": "",
                "stage_1_expected_num_trials": point[
                    "stage_1_expected_num_trials"
                ],
                "stage_2_expected_num_trials": point[
                    "stage_2_expected_num_trials"
                ],
                "top_m": privacy_points["two_stage_top_m"],
                "eta": privacy_points["two_stage_eta"],
                "epsilon": dp_result.epsilon,
                "delta": dp_result.delta,
                "best_renyi_order": dp_result.best_order,
                "is_at_min_order": dp_result.is_at_min_order,
                "is_at_max_order": dp_result.is_at_max_order,
                "min_renyi_order": int(orders[0]),
                "max_renyi_order": int(orders[-1]),
                "accounting_method": accounting_method,
                **accounting_metadata,
            }
        )

    paths = get_compilation_paths(exp_config)
    compiled_root = paths["compiled_root"]
    compiled_root.mkdir(parents=True, exist_ok=True)
    csv_path = compiled_root / "privacy_compute_results.csv"
    save_privacy_compute_rows(
        rows=privacy_rows,
        csv_path=csv_path,
    )

    method_labels = {
        "papernot_baseline": "Papernot baseline",
        "two_stage_tuning": "Two-stage tuning",
    }
    figure, axis = plt.subplots(figsize=(10, 5))
    for method in RESULT_FILENAMES:
        method_rows = sorted(
            (
                row
                for row in privacy_rows
                if row["method"] == method
            ),
            key=lambda row: row["expected_compute"],
        )
        axis.plot(
            [row["expected_compute"] for row in method_rows],
            [row["epsilon"] for row in method_rows],
            label=method_labels[method],
            marker="o",
        )

    axis.set_xlabel(
        "Expected compute (communication rounds × local updates)"
    )
    axis.set_ylabel(
        rf"$\varepsilon$ at $\delta={delta:.1e}$"
    )
    axis.set_title("Privacy-Compute Tradeoff")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    plot_path = compiled_root / "expected_compute_vs_privacy.png"
    figure.savefig(plot_path, dpi=300)
    plt.close(figure)

    return {
        "privacy_csv_path": csv_path,
        "plot_path": plot_path,
    }

def utility_compute_plot(config: DictConfig) -> None:
    exp_config = config.experiment

    if exp_config.run_mode.generate_stage_1_plan:
        E_K_values_N_stage = exp_config.base_E_K_list
        stage_compute_schedule = get_stage_compute_schedule(
            exp_config
        )
        fixed_compute = np.zeros_like(E_K_values_N_stage, dtype=float)
        E_K_values_papernot_baseline = np.zeros_like(E_K_values_N_stage, dtype=float)
        for i, E_K in enumerate(E_K_values_N_stage):
            fixed_compute[i] = calculate_compute_given_E_K_two_stage_tuning(exp_config, E_K, stage_compute_schedule)
            E_K_values_papernot_baseline[i] = calculate_E_K_given_compute_for_papernot(fixed_compute[i], stage_compute_schedule)
        generate_plan(
            exp_config,
            "papernot_baseline",
            1,
            E_K_values_papernot_baseline,
            exp_config.num_trials,
            exp_config.run_id,
            exp_config.hp_configuration_ids,
            PLAN_FILENAMES["papernot_baseline"][1],
        )
        generate_plan(
            exp_config,
            "two_stage_tuning",
            exp_config.n_stage_tuning.E_K_each_stage[0],
            E_K_values_N_stage,
            exp_config.num_trials,
            exp_config.run_id,
            exp_config.hp_configuration_ids,
            PLAN_FILENAMES["two_stage_tuning"][1],
        )


    if exp_config.run_mode.run_simulation_stage_1:
        run_planned_simulations(
            config,
            stage=1,
        )

    if exp_config.run_mode.generate_stage_2_plan:
        stage_1_plan = load_stage_1_plan(exp_config)
        stage_1_plan = map_stage_1_plan_runs(
            stage_1_plan,
            exp_config.hp_configuration_ids,
        )
        stage_1_plan = record_stage_1_run_scores(
            stage_1_plan,
            exp_config,
        )
        stage_1_plan = select_top_m_stage_1_runs(
            stage_1_plan,
            m=exp_config.n_stage_tuning.E_K_each_stage[0],
            mode=stage_1_plan["evaluation"][
                "selection_mode"
            ],
            seed=exp_config.seed,
        )
        generate_stage_2_plan(
            stage_1_plan,
            exp_config,
        )

    if exp_config.run_mode.run_simulation_stage_2:
        run_planned_simulations(
            config,
            stage=2,
        )

    if exp_config.run_mode.compile_result:
        compile_experiment_results(exp_config)
        plot_privacy_compute_plot(config)

EXPERIMENT_RUNNERS = {
    "utility_compute_plot": utility_compute_plot,
}


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config",
)
def main(config: DictConfig) -> None:
    runner_name = config.experiment.get("runner", "utility_compute_plot")

    try:
        runner = EXPERIMENT_RUNNERS[runner_name]
    except KeyError as error:
        available_runners = ", ".join(sorted(EXPERIMENT_RUNNERS))
        raise ValueError(
            f"Unknown experiment runner {runner_name!r}. "
            f"Available runners: {available_runners}"
        ) from error

    runner(config)


if __name__ == "__main__":
    main()
