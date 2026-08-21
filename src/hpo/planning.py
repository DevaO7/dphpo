"""Deterministic HPO trial-plan generation shared by all trainers."""

from collections import Counter
import copy
import json
import math
from pathlib import Path

import numpy as np

from privacy_accounting import dpsgd, rdp_utils, selection_accounting
from privacy_accounting.poisson import PoissonDistribution
from privacy_accounting.tnb import (
    TNBDistribution,
    _solve_gamma_for_conditional_mean,
)
from utils.hpo_config import get_two_stage_settings


PLAN_FILENAMES = {
    "papernot_baseline": {
        1: "papernot_baseline.JSON",
        2: "papernot_baseline.JSON",
    },
    "papernot_poisson_baseline": {
        1: "papernot_poisson_baseline.JSON",
        2: "papernot_poisson_baseline.JSON",
    },
    "two_stage_tuning": {
        1: "two_stage_tuning_stage_1.JSON",
        2: "two_stage_tuning_stage_2.JSON",
    },
    "two_stage_poisson_tuning": {
        1: "two_stage_poisson_tuning_stage_1.JSON",
        2: "two_stage_poisson_tuning_stage_2.JSON",
    },
}

PRIVACY_MATCHED_POINT_METADATA_FIELDS = (
    "target_epsilon",
    "achieved_epsilon",
    "delta",
    "noise_multiplier",
    "best_renyi_order",
    "privacy_calibration",
)

# UserAVG multiplies this seed by as much as 500 before passing it
# to NumPy's legacy uint32 RNG. Leave enough headroom for the round,
# local-step, and user offsets added during training.
MAX_TRAINING_BASE_SEED = 8_000_000

PAPERNOT_METHODS = {
    "papernot_baseline",
    "papernot_poisson_baseline",
}

TWO_STAGE_METHODS = {
    "two_stage_tuning",
    "two_stage_poisson_tuning",
}

POISSON_METHODS = {
    "papernot_poisson_baseline",
    "two_stage_poisson_tuning",
}


def _sampling_distribution_for_method(method):
    if method in POISSON_METHODS:
        return "poisson"
    return "tnb"


def _selection_method_for_method(method):
    if method == "papernot_poisson_baseline":
        return "papernot_poisson_top1"
    if method == "two_stage_poisson_tuning":
        return "papernot_poisson_top_m_then_top1"
    if method == "two_stage_tuning":
        return "papernot_top1"
    return "papernot_top1"


def _poisson_k_zero_fallback(config):
    poisson_config = config.get("poisson", {})
    seed = poisson_config.get("k_zero_model_seed", 2027)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(
            "poisson.k_zero_model_seed must be a non-negative integer."
        )
    return {
        "mechanism": "fixed_random_initialization",
        "model_seed": int(seed),
        "depends_on_private_data": False,
        "num_private_training_runs": 0,
        "description": (
            "When K=0, return the configured model architecture at a "
            "fixed random initialization without accessing private data."
        ),
    }


def get_selection_signature(config):
    """Return the immutable public-selection definition for central runs."""
    if "experiment" not in config or "run_settings" not in config:
        raise ValueError(
            "A selection signature requires the full experiment and "
            "run_settings configuration."
        )
    exp_config = config.experiment
    selection = exp_config.evaluation.selection
    utility = exp_config.evaluation.utility
    selection_metric = (
        str(selection.metric).strip().lower().replace(" ", "_")
    )
    selection_mode = str(selection.mode).strip().lower()
    if selection_metric not in {
        "validation_loss",
        "validation_accuracy",
    }:
        raise ValueError(
            "Peak-checkpoint experiments must select using "
            "validation_loss or validation_accuracy; got "
            f"{selection_metric!r}."
        )
    if selection_mode not in {"min", "max", "last_round"}:
        raise ValueError(
            "evaluation.selection.mode must be 'min', 'max', or "
            f"'last_round'; got {selection_mode!r}."
        )
    expected_selection_mode = (
        "min" if selection_metric.endswith("_loss") else "max"
    )
    if selection_mode not in {expected_selection_mode, "last_round"}:
        raise ValueError(
            f"{selection_metric} must use mode "
            f"{expected_selection_mode!r} or 'last_round'; got "
            f"{selection_mode!r}."
        )
    utility_at = str(utility.get("at", "selection_round")).strip().lower()
    if utility_at != "selected_checkpoint":
        raise ValueError(
            "Peak-checkpoint experiments require "
            "evaluation.utility.at='selected_checkpoint'."
        )
    configured_utility_metrics = utility.metrics
    if isinstance(configured_utility_metrics, str):
        configured_utility_metrics = [configured_utility_metrics]
    utility_metrics = [
        str(metric).strip().lower().replace(" ", "_")
        for metric in configured_utility_metrics
    ]
    if (
        not utility_metrics
        or len(set(utility_metrics)) != len(utility_metrics)
        or any(
            metric not in {"test_loss", "test_accuracy"}
            for metric in utility_metrics
        )
    ):
        raise ValueError(
            "Selected-checkpoint utility metrics must be a non-empty, "
            "unique subset of test_loss and test_accuracy."
        )
    evaluation_interval = config.run_settings.evaluation_interval
    if (
        isinstance(evaluation_interval, bool)
        or not isinstance(evaluation_interval, int)
        or evaluation_interval <= 0
    ):
        raise ValueError(
            "run_settings.evaluation_interval must be a positive integer."
        )
    split_config = exp_config.dataset.get("public_evaluation_split")
    if split_config is None:
        raise ValueError(
            "experiment.dataset.public_evaluation_split is required."
        )
    validation_fraction = float(
        split_config.get("validation_fraction", 0.5)
    )
    split_seed = split_config.get("seed", 0)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "public_evaluation_split.validation_fraction must be "
            "strictly between 0 and 1."
        )
    if (
        isinstance(split_seed, bool)
        or not isinstance(split_seed, int)
        or split_seed < 0
    ):
        raise ValueError(
            "public_evaluation_split.seed must be a non-negative integer."
        )
    peak_tie_break = str(
        selection.get("peak_tie_break", "earliest")
    ).strip().lower()
    if peak_tie_break != "earliest":
        raise ValueError(
            "evaluation.selection.peak_tie_break currently supports "
            "only 'earliest'."
        )
    return {
        "schema_version": 1,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "evaluation_interval": int(evaluation_interval),
        "peak_tie_break": peak_tie_break,
        "public_evaluation_split": {
            "source": "configured_test_split",
            "dataset_name": str(exp_config.dataset.name),
            "dataset_id": str(
                exp_config.dataset.get("dataset_id", "")
            ),
            "validation_fraction": validation_fraction,
            "seed": int(split_seed),
        },
        "utility_at": utility_at,
        "utility_metrics": utility_metrics,
        "peak_checkpoint_schema_version": 1,
    }


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


def _build_required_stage_run_specs(
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
                    "stage_1_run_directory": stage_1_run_directory,
                }
            )

            if include_stage_2:
                continuation_index = 0
                stage_2_base_seed = derive_stage_2_base_seed(
                    plan_seed=plan_seed,
                    hp_index=hp_id_to_index[hp_id],
                    stage_1_run_index=stage_1_run_index,
                    continuation_index=continuation_index,
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


def map_stage_1_plan_runs(
    stage_1_plan,
    hp_configuration_ids,
):
    """Map sampled HP occurrences to deterministic reusable run specs."""
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

    for point_index, point in enumerate(stage_1_plan["points"]):
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

            trial["sampled_stage_1_runs"] = sampled_stage_1_runs

    return stage_1_plan


def generate_plan(
    config,
    method,
    m,
    E_K_values,
    num_trials,
    run_id,
    hp_configuration_ids,
    plan_filename,
    *,
    point_metadata=None,
    plan_metadata=None,
    plan_directory=None,
):
    """Generate and persist a deterministic static HPO trial plan.

    ``point_metadata`` and ``plan_metadata`` allow experiment-specific
    planners to attach immutable metadata while retaining the shared
    sampling, seed derivation, and run-deduplication logic. The existing
    TNB methods use conditioned TNB sampling. The Poisson Papernot baseline
    uses an unconditioned count whose support includes zero. Poisson
    two-stage Stage 1 instead conditions on ``K >= m`` and calibrates the
    underlying Poisson rate so ``E[K | K >= m]`` equals ``E_K``.
    """
    if method not in PLAN_FILENAMES:
        raise ValueError(
            f"Unknown plan method {method!r}."
        )

    E_K_values = list(E_K_values)
    if point_metadata is None:
        point_metadata = [{} for _ in E_K_values]
    elif len(point_metadata) != len(E_K_values):
        raise ValueError(
            "point_metadata must contain one mapping per E_K value."
        )
    if plan_metadata is None:
        plan_metadata = {}

    hp_configuration_counts = {
        hp_id: 0
        for hp_id in hp_configuration_ids
    }
    required_hp_configuration_runs = {
        hp_id: 0
        for hp_id in hp_configuration_ids
    }
    points = []
    total_num_simulations = 0
    sampling_distribution = _sampling_distribution_for_method(method)
    if method == "papernot_poisson_baseline" and int(m) != 1:
        raise ValueError(
            "The Poisson Papernot baseline supports top-1 release only."
        )

    for point_index, E_K in enumerate(E_K_values):
        if method == "papernot_poisson_baseline":
            distribution = PoissonDistribution.from_mean(
                target_mean=E_K
            )
            gamma = None
        elif method == "two_stage_poisson_tuning":
            distribution = PoissonDistribution.from_conditional_mean(
                m=m,
                target_mean=E_K,
            )
            gamma = None
        else:
            gamma = _solve_gamma_for_conditional_mean(
                eta=config.eta,
                m=m,
                target_mean=E_K,
            )
            distribution = TNBDistribution(config.eta, gamma)
        trials = []

        for trial in range(num_trials):
            sampling_seed = trial + config.seed + point_index
            rng = np.random.default_rng(
                seed=sampling_seed
            )
            if method == "papernot_poisson_baseline":
                num_runs = int(distribution.sample(rng))
            elif method == "two_stage_poisson_tuning":
                num_runs = int(distribution.sample_conditional(m, rng))
            else:
                num_runs = int(distribution.sample_conditional(m, rng))
            sampled_hp_configuration_ids = (
                rng.choice(
                    hp_configuration_ids,
                    size=num_runs,
                    replace=True,
                ).tolist()
            )
            trials.append(
                {
                    "trial": trial,
                    "sampling_seed": int(sampling_seed),
                    "sampled_K": num_runs,
                    "sampled_hp_configuration_ids": (
                        sampled_hp_configuration_ids
                    ),
                }
            )
            total_num_simulations += num_runs
            current_hp_counts = Counter(
                sampled_hp_configuration_ids
            )

            for hp_id in hp_configuration_ids:
                current_count = current_hp_counts.get(hp_id, 0)
                hp_configuration_counts[hp_id] += current_count
                required_hp_configuration_runs[hp_id] = max(
                    required_hp_configuration_runs[hp_id],
                    current_count,
                )

        point = {
            "E_K": float(E_K),
            "trials": trials,
        }
        if method == "papernot_poisson_baseline":
            point["probability_K_zero"] = float(
                distribution.probability_zero()
            )
            point["poisson_rate"] = float(distribution.mu)
            point["expected_K_semantics"] = "unconditioned_mean"
        elif method == "two_stage_poisson_tuning":
            point.update(
                {
                    "poisson_rate": float(distribution.mu),
                    "conditioning_threshold": int(m),
                    "conditioning_probability": float(
                        distribution.survival_probability(m)
                    ),
                    "log_expected_binomial": float(
                        distribution.log_expected_binomial(m)
                    ),
                    "expected_K_semantics": "conditional_mean",
                }
            )
        else:
            point["gamma"] = float(gamma)
        metadata = dict(point_metadata[point_index])
        conflicting_keys = set(point).intersection(metadata)
        if conflicting_keys:
            raise ValueError(
                "point_metadata may not replace core plan fields: "
                f"{sorted(conflicting_keys)}."
            )
        point.update(metadata)
        points.append(point)

    total_num_required_runs = sum(
        required_hp_configuration_runs.values()
    )
    include_stage_2_specs = method in PAPERNOT_METHODS
    (
        required_stage_1_run_specs,
        required_stage_2_run_specs,
    ) = _build_required_stage_run_specs(
        required_hp_configuration_runs,
        hp_configuration_ids,
        include_stage_2=include_stage_2_specs,
        plan_seed=config.seed,
    )

    plan = {
        "method": method,
        "selection_method": _selection_method_for_method(method),
        "sampling_distribution": sampling_distribution,
        "stage_1_top_m": int(m),
        "eta": float(config.eta),
        "run_id": str(run_id),
        "plan_seed": int(config.seed),
        "hp_configuration_ids": [
            str(hp_id)
            for hp_id in hp_configuration_ids
        ],
        "points": points,
        "execution_summary": {
            "total_num_simulations": total_num_simulations,
            "hp_configuration_counts": hp_configuration_counts,
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
    if method == "papernot_poisson_baseline":
        plan["k_zero_fallback"] = _poisson_k_zero_fallback(config)
    conflicting_keys = set(plan).intersection(plan_metadata)
    if conflicting_keys:
        raise ValueError(
            "plan_metadata may not replace core plan fields: "
            f"{sorted(conflicting_keys)}."
        )
    plan.update(dict(plan_metadata))
    plan = map_stage_1_plan_runs(plan, hp_configuration_ids)

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

    if plan_directory is None:
        plan_directory = (
            Path(config.output.results_root)
            / str(config.name)
            / str(run_id)
            / "plan"
        )
    else:
        plan_directory = Path(plan_directory)
    plan_directory.mkdir(parents=True, exist_ok=True)
    plan_path = plan_directory / plan_filename

    temporary_path = plan_path.with_suffix(
        f"{plan_path.suffix}.tmp"
    )
    try:
        with temporary_path.open(mode="w", encoding="utf-8") as file:
            json.dump(plan, file, indent=4, allow_nan=False)
        temporary_path.replace(plan_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return plan_path


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


def load_stage_1_plan(
    config,
    plan_filename="two_stage_tuning_stage_1.JSON",
    plan_directory=None,
):
    if plan_directory is None:
        plan_directory = (
            Path(config.output.results_root)
            / str(config.name)
            / str(config.run_id)
            / "plan"
        )
    plan_path = Path(plan_directory) / plan_filename

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

def load_stage_2_plan(
    config,
    plan_filename="two_stage_tuning_stage_2.JSON",
    plan_directory=None,
):
    if plan_directory is None:
        plan_directory = (
            Path(config.output.results_root)
            / str(config.name)
            / str(config.run_id)
            / "plan"
        )
    plan_path = Path(plan_directory) / plan_filename
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


def load_simulation_plan(
    config,
    stage,
    *,
    selection_signature=None,
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

    configured_hp_ids = [
        str(hp_id)
        for hp_id in config.hp_configuration_ids
    ]
    metadata_checks = {
        "run_id": str(config.run_id),
        "plan_seed": int(config.seed),
        "hp_configuration_ids": configured_hp_ids,
        "eta": float(config.eta),
    }
    for key, expected_value in metadata_checks.items():
        if plan.get(key) != expected_value:
            raise ValueError(
                f"Simulation plan {plan_filename} has {key}="
                f"{plan.get(key)!r}, but the experiment configuration "
                f"requires {expected_value!r}. Regenerate the plan or "
                "restore the matching experiment configuration."
            )
    if selection_signature is not None:
        if plan.get("plan_type") != "compute_matched":
            raise ValueError(
                f"Simulation plan {plan_filename} must have "
                "plan_type='compute_matched'. Regenerate the plan."
            )
        if plan.get("selection_signature") != selection_signature:
            raise ValueError(
                f"Simulation plan {plan_filename} has an incompatible "
                "selection signature. Regenerate the plan."
            )
    expected_sampling_distribution = _sampling_distribution_for_method(
        method
    )
    observed_sampling_distribution = plan.get(
        "sampling_distribution",
        "tnb",
    )
    if observed_sampling_distribution != expected_sampling_distribution:
        raise ValueError(
            f"Simulation plan {plan_filename} uses "
            f"sampling_distribution={observed_sampling_distribution!r}, "
            f"expected {expected_sampling_distribution!r}. Regenerate "
            "the plan."
        )
    if method == "papernot_poisson_baseline":
        if plan.get("selection_method") != "papernot_poisson_top1":
            raise ValueError(
                f"Simulation plan {plan_filename} must use "
                "selection_method='papernot_poisson_top1'."
            )
        if plan.get("k_zero_fallback") != _poisson_k_zero_fallback(config):
            raise ValueError(
                f"Simulation plan {plan_filename} has an invalid K=0 "
                "fallback definition. Regenerate the plan."
            )
    elif method == "two_stage_poisson_tuning":
        if plan.get("selection_method") != (
            "papernot_poisson_top_m_then_top1"
        ):
            raise ValueError(
                f"Simulation plan {plan_filename} must use the Poisson "
                "two-stage selection method. Regenerate the plan."
            )
        if stage == 2 and plan.get("stage_2_k_zero_fallback") != (
            _poisson_k_zero_fallback(config)
        ):
            raise ValueError(
                f"Simulation plan {plan_filename} has an invalid "
                "Stage-2 K=0 fallback definition. Regenerate the plan."
            )

    configured_num_trials = int(config.num_trials)
    points = plan.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError(
            f"Simulation plan {plan_filename} must contain a non-empty "
            "points list."
        )
    for point_index, point in enumerate(points):
        trials = point.get("trials") if isinstance(point, dict) else None
        if (
            not isinstance(trials, list)
            or len(trials) != configured_num_trials
        ):
            raise ValueError(
                f"Simulation plan {plan_filename} point {point_index} "
                f"contains {len(trials) if isinstance(trials, list) else 'an invalid number of'} "
                f"trials, but experiment.num_trials is "
                f"{configured_num_trials}. Regenerate the plan."
            )

    configured_point_count = len(config.base_E_K_list)
    if len(points) != configured_point_count:
        raise ValueError(
            f"Simulation plan {plan_filename} contains {len(points)} "
            "expected-trial points, but experiment.base_E_K_list "
            f"contains {configured_point_count}. Regenerate the plan."
        )

    if method in TWO_STAGE_METHODS:
        two_stage_settings = get_two_stage_settings(config)
        if plan.get("stage_1_top_m") != (
            two_stage_settings.num_survivors
        ):
            raise ValueError(
                f"Simulation plan {plan_filename} has stage_1_top_m="
                f"{plan.get('stage_1_top_m')!r}, but the experiment "
                "configuration requires "
                f"{two_stage_settings.num_survivors}. Regenerate the "
                "plan."
            )

        expected_stage_1_values = np.asarray(
            [float(value) for value in config.base_E_K_list],
            dtype=float,
        )
        point_key = "E_K" if stage == 1 else "stage_1_E_K"
        try:
            observed_stage_1_values = np.asarray(
                [float(point[point_key]) for point in points],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Simulation plan {plan_filename} has invalid "
                f"{point_key!r} values. Regenerate the plan."
            ) from error
        if not np.array_equal(
            observed_stage_1_values,
            expected_stage_1_values,
        ):
            raise ValueError(
                f"Simulation plan {plan_filename} expected-trial grid "
                "does not match experiment.base_E_K_list. Regenerate "
                "the plan."
            )

        if stage == 2:
            try:
                observed_stage_2_values = np.asarray(
                    [float(point["stage_2_E_K"]) for point in points],
                    dtype=float,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Simulation plan {plan_filename} has invalid "
                    "'stage_2_E_K' values. Regenerate the plan."
                ) from error
            expected_stage_2_values = np.full(
                len(points),
                two_stage_settings.stage_2_expected_trials,
                dtype=float,
            )
            if not np.array_equal(
                observed_stage_2_values,
                expected_stage_2_values,
            ):
                raise ValueError(
                    f"Simulation plan {plan_filename} does not match "
                    "two_stage.stage_2_expected_trials. Regenerate the "
                    "plan."
                )
        if method == "two_stage_poisson_tuning":
            for point_index, (point, expected_stage_1_k) in enumerate(
                zip(points, expected_stage_1_values)
            ):
                stage_1_distribution = (
                    PoissonDistribution.from_conditional_mean(
                        m=two_stage_settings.num_survivors,
                        target_mean=expected_stage_1_k,
                    )
                )
                rate_key = (
                    "poisson_rate"
                    if stage == 1
                    else "stage_1_poisson_rate"
                )
                if not math.isclose(
                    float(point.get(rate_key, math.nan)),
                    stage_1_distribution.mu,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"Simulation plan {plan_filename} point "
                        f"{point_index} has an invalid {rate_key}. "
                        "Regenerate the plan."
                    )
                trial_container_key = (
                    None if stage == 1 else "trial_stage_2"
                )
                minimum_k = (
                    two_stage_settings.num_survivors
                    if stage == 1
                    else 0
                )
                for trial_index, trial in enumerate(point["trials"]):
                    sampled_trial = (
                        trial
                        if trial_container_key is None
                        else trial[trial_container_key]
                    )
                    sampled_k = sampled_trial.get("sampled_K")
                    if (
                        isinstance(sampled_k, bool)
                        or not isinstance(sampled_k, int)
                        or sampled_k < minimum_k
                    ):
                        raise ValueError(
                            f"Simulation plan {plan_filename} point "
                            f"{point_index}, trial {trial_index} has "
                            f"invalid sampled_K={sampled_k!r}."
                        )
                    if stage == 2 and len(
                        sampled_trial.get("sampled_stage_2_runs", [])
                    ) != sampled_k:
                        raise ValueError(
                            f"Simulation plan {plan_filename} point "
                            f"{point_index}, trial {trial_index} has "
                            "Stage-2 run entries inconsistent with "
                            "sampled_K."
                        )
                if stage == 2:
                    expected_stage_2_k = (
                        two_stage_settings.stage_2_expected_trials
                    )
                    if not math.isclose(
                        float(
                            point.get(
                                "stage_2_poisson_rate",
                                math.nan,
                            )
                        ),
                        expected_stage_2_k,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ) or not math.isclose(
                        float(
                            point.get(
                                "stage_2_probability_K_zero",
                                math.nan,
                            )
                        ),
                        math.exp(-expected_stage_2_k),
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    ):
                        raise ValueError(
                            f"Simulation plan {plan_filename} point "
                            f"{point_index} has invalid Stage-2 Poisson "
                            "metadata. Regenerate the plan."
                        )
    else:
        if plan.get("stage_1_top_m") != 1:
            raise ValueError(
                f"Simulation plan {plan_filename} must have "
                "stage_1_top_m=1. Regenerate the plan."
            )
        stage_1_compute = int(config.simulation.stage_1_end)
        stage_2_end = int(config.simulation.stage_2_end)
        stage_2_compute = stage_2_end - stage_1_compute
        two_stage_settings = get_two_stage_settings(config)
        expected_papernot_values = np.asarray(
            [
                (
                    float(value) * stage_1_compute
                    + two_stage_settings.stage_2_expected_trials
                    * stage_2_compute
                )
                / stage_2_end
                for value in config.base_E_K_list
            ],
            dtype=float,
        )
        try:
            observed_papernot_values = np.asarray(
                [float(point["E_K"]) for point in points],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Simulation plan {plan_filename} has invalid E_K values."
            ) from error
        if not np.allclose(
            observed_papernot_values,
            expected_papernot_values,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                f"Simulation plan {plan_filename} does not use the "
                "compute-matched Papernot expected-trial grid. Regenerate "
                "the plan."
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


def _get_run_spec_context(plan, stage):
    plan_hp_ids = plan.get("hp_configuration_ids")
    if (
        not isinstance(plan_hp_ids, list)
        or not plan_hp_ids
        or len(plan_hp_ids) != len(set(plan_hp_ids))
    ):
        raise ValueError(
            f"The Stage-{stage} plan must contain a non-empty, "
            "unique hp_configuration_ids list. Regenerate the plan."
        )

    hp_id_to_index = {
        str(hp_id): index
        for index, hp_id in enumerate(plan_hp_ids)
    }
    plan_seed = plan.get("plan_seed")
    if (
        isinstance(plan_seed, bool)
        or not isinstance(plan_seed, int)
        or plan_seed < 0
    ):
        raise ValueError(
            f"Stage-{stage} plan_seed must be a non-negative integer."
        )

    return hp_id_to_index, plan_seed


def get_required_stage_1_run_specs(
    stage_1_plan,
    hp_configuration_id,
):
    """Validate and select Stage-1 run specs for one HP identity."""
    hp_configuration_id = str(hp_configuration_id)
    hp_id_to_index, plan_seed = _get_run_spec_context(
        stage_1_plan,
        stage=1,
    )
    required_keys = {
        "hp_configuration_id",
        "stage_1_run_index",
        "stage_1_base_seed",
        "stage_1_run_directory",
    }
    selected_specs = []
    observed_base_seeds = set()

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
        run_index = spec["stage_1_run_index"]
        base_seed = spec["stage_1_base_seed"]
        if spec_hp_id not in hp_id_to_index:
            raise ValueError(
                f"Required Stage-1 run entry {spec_index} contains "
                f"unknown HP configuration {spec_hp_id!r}."
            )
        if (
            isinstance(run_index, bool)
            or not isinstance(run_index, int)
            or run_index < 0
            or isinstance(base_seed, bool)
            or not isinstance(base_seed, int)
            or not 0 <= base_seed < MAX_TRAINING_BASE_SEED
        ):
            raise ValueError(
                f"Required Stage-1 run entry {spec_index} has "
                "invalid seed values."
            )
        if base_seed in observed_base_seeds:
            raise ValueError(
                "Stage-1 base seeds must be unique; "
                f"{base_seed} is repeated."
            )
        observed_base_seeds.add(base_seed)

        expected_seed = derive_stage_1_base_seed(
            plan_seed=plan_seed,
            hp_index=hp_id_to_index[spec_hp_id],
            stage_1_run_index=run_index,
        )
        if base_seed != expected_seed:
            raise ValueError(
                f"Required Stage-1 run entry {spec_index} has an "
                "inconsistent Stage-1 base seed."
            )

        run_directory = Path(str(spec["stage_1_run_directory"]))
        expected_directory = Path(spec_hp_id, f"run_{run_index}")
        if (
            run_directory.is_absolute()
            or ".." in run_directory.parts
            or run_directory != expected_directory
        ):
            raise ValueError(
                f"Required Stage-1 run entry {spec_index} has "
                "inconsistent or unsafe path metadata."
            )
        if spec_hp_id == hp_configuration_id:
            selected_specs.append(dict(spec))

    if not selected_specs:
        # Random-count plans may legitimately assign no work to a known HP,
        # and an all-zero Poisson plan assigns no work to any HP.
        return []
    return selected_specs


def get_required_stage_2_run_specs(
    stage_2_plan,
    hp_configuration_id,
):
    """Validate and select Stage-2 run specs for one HP identity."""
    hp_configuration_id = str(hp_configuration_id)
    hp_id_to_index, plan_seed = _get_run_spec_context(
        stage_2_plan,
        stage=2,
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
    observed_base_seeds = set()

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
        run_index = spec["stage_1_run_index"]
        stage_1_seed = spec["stage_1_base_seed"]
        continuation_index = spec["continuation_index"]
        stage_2_seed = spec["stage_2_base_seed"]
        integer_values = (
            run_index,
            stage_1_seed,
            continuation_index,
            stage_2_seed,
        )
        if spec_hp_id not in hp_id_to_index:
            raise ValueError(
                f"Required Stage-2 run entry {spec_index} contains "
                f"unknown HP configuration {spec_hp_id!r}."
            )
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in integer_values
            )
            or stage_1_seed >= MAX_TRAINING_BASE_SEED
            or stage_2_seed >= MAX_TRAINING_BASE_SEED
        ):
            raise ValueError(
                f"Required Stage-2 run entry {spec_index} has "
                "invalid Stage-1 or Stage-2 seed values."
            )
        if stage_2_seed in observed_base_seeds:
            raise ValueError(
                "Stage-2 base seeds must be unique; "
                f"{stage_2_seed} is repeated."
            )
        observed_base_seeds.add(stage_2_seed)

        expected_stage_1_seed = derive_stage_1_base_seed(
            plan_seed=plan_seed,
            hp_index=hp_id_to_index[spec_hp_id],
            stage_1_run_index=run_index,
        )
        expected_stage_2_seed = derive_stage_2_base_seed(
            plan_seed=plan_seed,
            hp_index=hp_id_to_index[spec_hp_id],
            stage_1_run_index=run_index,
            continuation_index=continuation_index,
        )
        if stage_1_seed != expected_stage_1_seed:
            raise ValueError(
                f"Required Stage-2 run entry {spec_index} has an "
                "inconsistent Stage-1 base seed."
            )
        if stage_2_seed != expected_stage_2_seed:
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
        expected_stage_1_directory = Path(
            spec_hp_id,
            f"run_{run_index}",
        )
        expected_stage_2_directory = (
            expected_stage_1_directory
            / f"continuation_{continuation_index}"
        )
        if (
            stage_1_directory.is_absolute()
            or stage_2_directory.is_absolute()
            or ".." in stage_1_directory.parts
            or ".." in stage_2_directory.parts
            or stage_1_directory != expected_stage_1_directory
            or stage_2_directory != expected_stage_2_directory
        ):
            raise ValueError(
                f"Required Stage-2 run entry {spec_index} has "
                "inconsistent or unsafe path metadata."
            )
        if spec_hp_id == hp_configuration_id:
            selected_specs.append(dict(spec))

    if not selected_specs:
        # Treat a known HP with no sampled appearances as a completed no-op.
        return []
    return selected_specs


def get_required_simulation_run_specs(
    plan,
    stage,
    hp_configuration_id,
):
    """Select validated run specs for either simulation stage."""
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

def build_stage_2_plan(
    stage_1_plan,
    config,
):
    method = stage_1_plan.get("method")
    if method not in TWO_STAGE_METHODS:
        raise ValueError(
            "A two-stage Stage-2 plan must be built from a "
            f"recognized two-stage Stage-1 plan; got {method!r}."
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
    two_stage_settings = get_two_stage_settings(config)
    if stage_1_top_m != two_stage_settings.num_survivors:
        raise ValueError(
            "The Stage-1 plan survivor count does not match "
            "two_stage.num_survivors: "
            f"plan={stage_1_top_m}, "
            f"config={two_stage_settings.num_survivors}."
        )
    stage_2_expected_k = (
        two_stage_settings.stage_2_expected_trials
    )

    sampling_distribution = _sampling_distribution_for_method(method)
    if sampling_distribution == "poisson":
        stage_2_distribution = PoissonDistribution.from_mean(
            target_mean=stage_2_expected_k
        )
        stage_2_gamma = None
    else:
        stage_2_gamma = _solve_gamma_for_conditional_mean(
            eta=config.eta,
            m=1,
            target_mean=stage_2_expected_k,
        )
        stage_2_distribution = TNBDistribution(
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
            if sampling_distribution == "poisson":
                sampled_k = int(stage_2_distribution.sample(rng))
            else:
                sampled_k = int(
                    stage_2_distribution.sample_conditional(1, rng)
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

        stage_2_point = {
            "stage_1_E_K": float(stage_1_point["E_K"]),
            "stage_2_E_K": stage_2_expected_k,
            "trials": stage_2_trials,
        }
        if sampling_distribution == "poisson":
            stage_2_point.update(
                {
                    "stage_1_poisson_rate": float(
                        stage_1_point["poisson_rate"]
                    ),
                    "stage_1_conditioning_threshold": int(
                        stage_1_point["conditioning_threshold"]
                    ),
                    "stage_1_conditioning_probability": float(
                        stage_1_point["conditioning_probability"]
                    ),
                    "stage_1_log_expected_binomial": float(
                        stage_1_point["log_expected_binomial"]
                    ),
                    "stage_2_poisson_rate": float(
                        stage_2_distribution.mu
                    ),
                    "stage_2_probability_K_zero": float(
                        stage_2_distribution.probability_zero()
                    ),
                    "stage_1_expected_K_semantics": (
                        "conditional_mean"
                    ),
                    "stage_2_expected_K_semantics": (
                        "unconditioned_mean"
                    ),
                }
            )
        else:
            stage_2_point.update(
                {
                    "stage_1_gamma": float(stage_1_point["gamma"]),
                    "stage_2_gamma": float(stage_2_gamma),
                }
            )
        for key in PRIVACY_MATCHED_POINT_METADATA_FIELDS:
            if key in stage_1_point:
                stage_2_point[key] = copy.deepcopy(
                    stage_1_point[key]
                )
        points.append(stage_2_point)

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

    stage_2_plan = {
        "method": method,
        "selection_method": _selection_method_for_method(method),
        "sampling_distribution": sampling_distribution,
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
    if "plan_type" in stage_1_plan:
        stage_2_plan["plan_type"] = stage_1_plan["plan_type"]
    if "selection_signature" in stage_1_plan:
        stage_2_plan["selection_signature"] = copy.deepcopy(
            stage_1_plan["selection_signature"]
        )
    if sampling_distribution == "poisson":
        stage_2_plan["stage_2_k_zero_fallback"] = (
            _poisson_k_zero_fallback(config)
        )
    return stage_2_plan

def generate_stage_2_plan(
    stage_1_plan,
    config,
    plan_filename=None,
    *,
    plan_directory=None,
):
    stage_2_plan = build_stage_2_plan(
        stage_1_plan,
        config,
    )
    if plan_filename is None:
        plan_filename = PLAN_FILENAMES[stage_2_plan["method"]][2]
    if plan_directory is None:
        plan_directory = (
            Path(config.output.results_root)
            / str(config.name)
            / str(config.run_id)
            / "plan"
        )
    else:
        plan_directory = Path(plan_directory)
    plan_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    plan_path = plan_directory / plan_filename

    temporary_path = plan_path.with_suffix(
        f"{plan_path.suffix}.tmp"
    )
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

            for chunk in encoder.iterencode(stage_2_plan):
                file.write(chunk)
                pending_characters += len(chunk)

                if pending_characters >= 1_000_000:
                    file.flush()
                    pending_characters = 0
        temporary_path.replace(plan_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return plan_path


def _validate_positive_finite(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _privacy_calibration_settings(config, target_epsilon):
    """Validate and return shared central DP-SGD calibration settings."""
    exp_config = config.experiment
    target_epsilon = _validate_positive_finite(
        target_epsilon,
        "target_epsilon",
    )
    delta = float(exp_config.privacy.delta)
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("privacy.delta must satisfy 0 < delta < 1.")

    max_renyi_order = int(exp_config.privacy.max_renyi_order)
    if max_renyi_order < 3:
        raise ValueError(
            "privacy.max_renyi_order must be at least 3."
        )

    stage_1_end = int(exp_config.simulation.stage_1_end)
    stage_2_end = int(exp_config.simulation.stage_2_end)
    if stage_1_end <= 0 or stage_2_end <= stage_1_end:
        raise ValueError(
            "Privacy calibration requires 0 < stage_1_end < "
            "stage_2_end."
        )

    if not bool(config.run_settings.dp):
        raise ValueError(
            "Privacy calibration requires run_settings.dp=true."
        )
    if (
        str(config.run_settings.data_sampling_scheme)
        != "poisson_sampling"
    ):
        raise ValueError(
            "Central DP-SGD privacy calibration requires "
            "data_sampling_scheme='poisson_sampling'."
        )

    sampling_rate = float(config.run_settings.sampling_rate)
    if (
        not math.isfinite(sampling_rate)
        or not 0.0 < sampling_rate <= 1.0
    ):
        raise ValueError(
            "run_settings.sampling_rate must satisfy 0 < rate <= 1."
        )

    search_config = exp_config.privacy.get("sigma_search", {})
    initial_sigma = _validate_positive_finite(
        search_config.get(
            "initial_sigma",
            config.run_settings.noise_multiplier,
        ),
        "privacy.sigma_search.initial_sigma",
    )
    minimum_sigma = _validate_positive_finite(
        search_config.get("minimum_sigma", 1e-3),
        "privacy.sigma_search.minimum_sigma",
    )
    maximum_sigma = _validate_positive_finite(
        search_config.get("maximum_sigma", 1e3),
        "privacy.sigma_search.maximum_sigma",
    )
    if not minimum_sigma < maximum_sigma:
        raise ValueError(
            "privacy.sigma_search.minimum_sigma must be smaller "
            "than maximum_sigma."
        )
    if not minimum_sigma <= initial_sigma <= maximum_sigma:
        raise ValueError(
            "privacy.sigma_search.initial_sigma must lie within the "
            "configured search interval."
        )

    relative_tolerance = _validate_positive_finite(
        search_config.get("relative_tolerance", 1e-6),
        "privacy.sigma_search.relative_tolerance",
    )
    if relative_tolerance >= 1.0:
        raise ValueError(
            "privacy.sigma_search.relative_tolerance must be less "
            "than 1."
        )
    raw_max_iterations = search_config.get("max_iterations", 80)
    if isinstance(raw_max_iterations, bool):
        raise ValueError(
            "privacy.sigma_search.max_iterations must be a positive "
            "integer."
        )
    max_iterations = int(raw_max_iterations)
    if max_iterations <= 0 or max_iterations != raw_max_iterations:
        raise ValueError(
            "privacy.sigma_search.max_iterations must be a positive "
            "integer."
        )

    return {
        "target_epsilon": target_epsilon,
        "delta": delta,
        "orders": np.arange(2, max_renyi_order + 1),
        "min_renyi_order": 2,
        "max_renyi_order": max_renyi_order,
        "stage_1_num_rounds": stage_1_end,
        "stage_2_num_rounds": stage_2_end - stage_1_end,
        "sampling_rate": sampling_rate,
        "initial_sigma": initial_sigma,
        "minimum_sigma": minimum_sigma,
        "maximum_sigma": maximum_sigma,
        "relative_tolerance": relative_tolerance,
        "max_iterations": max_iterations,
    }


def _selection_result_for_noise_multiplier(
    config,
    method,
    expected_num_trials,
    noise_multiplier,
    settings,
):
    """Return the end-to-end HPO RDP result for one candidate sigma."""
    common_config = {
        "data_sampling_rate": settings["sampling_rate"],
        "sigma_gaussian": float(noise_multiplier),
    }
    stage_1_curve = dpsgd.compute_dpsgd_rdp(
        config={
            **common_config,
            "num_rounds": settings["stage_1_num_rounds"],
        },
        orders=settings["orders"],
    )
    stage_2_curve = dpsgd.compute_dpsgd_rdp(
        config={
            **common_config,
            "num_rounds": settings["stage_2_num_rounds"],
        },
        orders=settings["orders"],
    )

    eta = float(config.experiment.eta)
    if method in PAPERNOT_METHODS:
        complete_base_curve = rdp_utils.compose_rdp_curves(
            stage_1_curve,
            stage_2_curve,
        )
    if method == "papernot_baseline":
        selection_result = selection_accounting.compute_top1_rdp(
            base_rdp_curve=complete_base_curve,
            expected_num_trials=expected_num_trials,
            eta=eta,
        )
    elif method == "papernot_poisson_baseline":
        selection_result = (
            selection_accounting.compute_top1_rdp_poisson(
                base_rdp_curve=complete_base_curve,
                expected_num_trials=expected_num_trials,
            )
        )
    elif method == "two_stage_tuning":
        two_stage_settings = get_two_stage_settings(config.experiment)
        selection_result = selection_accounting.compute_two_stage_rdp(
            stage_1_base_rdp_curve=stage_1_curve,
            stage_2_base_rdp_curve=stage_2_curve,
            m=two_stage_settings.num_survivors,
            expected_num_trials_stage_1=expected_num_trials,
            expected_num_trials_stage_2=(
                two_stage_settings.stage_2_expected_trials
            ),
            eta_stage_1=eta,
            eta_stage_2=eta,
        )
    elif method == "two_stage_poisson_tuning":
        two_stage_settings = get_two_stage_settings(config.experiment)
        selection_result = (
            selection_accounting.compute_two_stage_rdp_poisson(
                stage_1_base_rdp_curve=stage_1_curve,
                stage_2_base_rdp_curve=stage_2_curve,
                m=two_stage_settings.num_survivors,
                expected_num_trials_stage_1=expected_num_trials,
                expected_num_trials_stage_2=(
                    two_stage_settings.stage_2_expected_trials
                ),
            )
        )
    else:
        raise ValueError(f"Unknown privacy-calibration method {method!r}.")

    return selection_result


def _epsilon_for_noise_multiplier(
    config,
    method,
    expected_num_trials,
    noise_multiplier,
    settings,
):
    """Evaluate end-to-end HPO epsilon for one candidate sigma."""
    selection_result = _selection_result_for_noise_multiplier(
        config=config,
        method=method,
        expected_num_trials=expected_num_trials,
        noise_multiplier=noise_multiplier,
        settings=settings,
    )

    return rdp_utils.convert_rdp_to_approx_dp(
        selection_result.rdp_curve,
        delta=settings["delta"],
    )


def _calibrate_noise_multiplier(
    config,
    method,
    target_epsilon,
    expected_num_trials,
):
    """Find the smallest feasible sigma using bracketed log bisection."""
    if method not in PLAN_FILENAMES:
        raise ValueError(f"Unknown plan method {method!r}.")
    expected_num_trials = _validate_positive_finite(
        expected_num_trials,
        "expected_num_trials",
    )
    settings = _privacy_calibration_settings(
        config,
        target_epsilon,
    )
    target_epsilon = settings["target_epsilon"]
    evaluations = 0

    def evaluate(sigma):
        nonlocal evaluations
        evaluations += 1
        return _epsilon_for_noise_multiplier(
            config=config,
            method=method,
            expected_num_trials=expected_num_trials,
            noise_multiplier=sigma,
            settings=settings,
        )

    initial_sigma = settings["initial_sigma"]
    initial_result = evaluate(initial_sigma)

    if initial_result.epsilon > target_epsilon:
        lower_sigma = initial_sigma
        lower_result = initial_result
        upper_sigma = initial_sigma
        upper_result = initial_result
        while upper_result.epsilon > target_epsilon:
            lower_sigma = upper_sigma
            lower_result = upper_result
            upper_sigma = min(
                2.0 * upper_sigma,
                settings["maximum_sigma"],
            )
            if upper_sigma == lower_sigma:
                raise ValueError(
                    "Could not satisfy target_epsilon within the "
                    "configured maximum_sigma."
                )
            upper_result = evaluate(upper_sigma)
    else:
        upper_sigma = initial_sigma
        upper_result = initial_result
        lower_sigma = initial_sigma
        lower_result = initial_result
        while lower_result.epsilon <= target_epsilon:
            upper_sigma = lower_sigma
            upper_result = lower_result
            lower_sigma = max(
                0.5 * lower_sigma,
                settings["minimum_sigma"],
            )
            if lower_sigma == upper_sigma:
                raise ValueError(
                    "Could not bracket target_epsilon within the "
                    "configured minimum_sigma."
                )
            lower_result = evaluate(lower_sigma)

    if not (
        lower_result.epsilon > target_epsilon
        and upper_result.epsilon <= target_epsilon
    ):
        raise RuntimeError(
            "Noise calibration failed to construct a valid bracket."
        )

    iterations = 0
    for iterations in range(1, settings["max_iterations"] + 1):
        if (
            (upper_sigma - lower_sigma) / upper_sigma
            <= settings["relative_tolerance"]
        ):
            break
        midpoint_sigma = math.sqrt(lower_sigma * upper_sigma)
        midpoint_result = evaluate(midpoint_sigma)
        if midpoint_result.epsilon > target_epsilon:
            lower_sigma = midpoint_sigma
            lower_result = midpoint_result
        else:
            upper_sigma = midpoint_sigma
            upper_result = midpoint_result
    else:
        raise RuntimeError(
            "Noise calibration did not converge within max_iterations."
        )

    if upper_result.is_at_min_order or upper_result.is_at_max_order:
        boundary = (
            "minimum"
            if upper_result.is_at_min_order
            else "maximum"
        )
        raise RuntimeError(
            "The calibrated privacy result uses the "
            f"{boundary} configured Renyi order "
            f"({upper_result.best_order}). Expand the order range."
        )

    calibration = {
        "method": method,
        "target_epsilon": target_epsilon,
        "achieved_epsilon": float(upper_result.epsilon),
        "delta": float(upper_result.delta),
        "noise_multiplier": float(upper_sigma),
        "best_renyi_order": float(upper_result.best_order),
        "min_renyi_order": settings["min_renyi_order"],
        "max_renyi_order": settings["max_renyi_order"],
        "relative_sigma_tolerance": settings["relative_tolerance"],
        "bisection_iterations": iterations,
        "accountant_evaluations": evaluations,
        "accounting_method": "numerical",
    }
    if method == "papernot_poisson_baseline":
        selection_result = _selection_result_for_noise_multiplier(
            config=config,
            method=method,
            expected_num_trials=expected_num_trials,
            noise_multiplier=upper_sigma,
            settings=settings,
        )
        target_index = int(
            np.flatnonzero(
                np.isclose(
                    selection_result.rdp_curve.orders,
                    upper_result.best_order,
                    rtol=0.0,
                    atol=1e-12,
                )
            )[0]
        )
        source_order = float(
            selection_result.envelope_source_orders[target_index]
        )
        source_index = int(
            np.flatnonzero(
                np.isclose(
                    selection_result.raw_rdp_curve.orders,
                    source_order,
                    rtol=0.0,
                    atol=1e-12,
                )
            )[0]
        )
        calibration["poisson_theorem_6"] = {
            "poisson_mean": float(expected_num_trials),
            "output_renyi_order": float(upper_result.best_order),
            "theorem_source_order": source_order,
            "base_rdp_epsilon": float(
                selection_result.base_rdp_curve.epsilons[source_index]
            ),
            "hat_epsilon": float(
                selection_result.hat_epsilons[source_index]
            ),
            "hat_delta": float(
                selection_result.hat_deltas[source_index]
            ),
            "best_auxiliary_renyi_order": float(
                selection_result.best_auxiliary_orders[source_index]
            ),
            "raw_theorem_rdp_epsilon": float(
                selection_result.raw_rdp_curve.epsilons[source_index]
            ),
            "enveloped_rdp_epsilon": float(
                selection_result.rdp_curve.epsilons[target_index]
            ),
            "final_conversion_delta": float(upper_result.delta),
            "rdp_to_dp_exponent_sign": (
                "base_rdp_epsilon_minus_hat_epsilon"
            ),
        }
    elif method == "two_stage_poisson_tuning":
        selection_result = _selection_result_for_noise_multiplier(
            config=config,
            method=method,
            expected_num_trials=expected_num_trials,
            noise_multiplier=upper_sigma,
            settings=settings,
        )
        stage_1_distribution = selection_result.stage_1.distribution
        stage_2_distribution = selection_result.stage_2.distribution
        calibration["poisson_two_stage"] = {
            "stage_1_expected_num_trials": float(expected_num_trials),
            "stage_1_poisson_rate": float(stage_1_distribution.mu),
            "stage_1_conditioning_threshold": int(
                selection_result.stage_1.m
            ),
            "stage_1_conditioning_probability": float(
                stage_1_distribution.survival_probability(
                    selection_result.stage_1.m
                )
            ),
            "stage_1_log_expected_binomial": float(
                selection_result.stage_1.log_expected_binomial
            ),
            "stage_2_expected_num_trials": float(
                stage_2_distribution.mu
            ),
            "stage_2_poisson_rate": float(stage_2_distribution.mu),
            "stage_2_probability_K_zero": float(
                stage_2_distribution.probability_zero()
            ),
            "stage_1_conditioning": "K1 >= m",
            "stage_2_conditioning": "none",
        }
    return calibration


def get_sigma_for_target_epsilon_papernot(
    config,
    target_epsilon,
    E_k,
):
    """Return Papernot's calibrated sigma for one (epsilon, E[K])."""
    return _calibrate_noise_multiplier(
        config=config,
        method="papernot_baseline",
        target_epsilon=target_epsilon,
        expected_num_trials=E_k,
    )["noise_multiplier"]


def get_sigma_for_target_epsilon_papernot_poisson(
    config,
    target_epsilon,
    E_k,
):
    """Return Poisson Papernot's sigma for one (epsilon, E[K])."""
    return _calibrate_noise_multiplier(
        config=config,
        method="papernot_poisson_baseline",
        target_epsilon=target_epsilon,
        expected_num_trials=E_k,
    )["noise_multiplier"]


def get_sigma_for_target_epsilon_two_stage(
    config,
    target_epsilon,
    E_k,
):
    """Return two-stage calibrated sigma for one (epsilon, E[K1])."""
    return _calibrate_noise_multiplier(
        config=config,
        method="two_stage_tuning",
        target_epsilon=target_epsilon,
        expected_num_trials=E_k,
    )["noise_multiplier"]


def get_sigma_for_target_epsilon_two_stage_poisson(
    config,
    target_epsilon,
    E_k,
):
    """Return Poisson two-stage sigma for one (epsilon, E[K1])."""
    return _calibrate_noise_multiplier(
        config=config,
        method="two_stage_poisson_tuning",
        target_epsilon=target_epsilon,
        expected_num_trials=E_k,
    )["noise_multiplier"]


def _path_value_slug(value, name):
    value = _validate_positive_finite(value, name)
    text = np.format_float_positional(value, trim="-")
    return text.replace(".", "p")


def get_privacy_matched_value_slug(value, name="value"):
    """Return the canonical path slug used for privacy coordinates."""
    return _path_value_slug(value, name)


def get_privacy_matched_plan_directory(
    config,
    method,
    target_epsilon,
    E_k,
):
    """Return the plan directory for one privacy-matched cell."""
    exp_config = config.experiment
    return (
        Path(exp_config.output.results_root)
        / str(exp_config.name)
        / str(exp_config.run_id)
        / "plan"
        / get_privacy_matched_point_relative_directory(
            method=method,
            target_epsilon=target_epsilon,
            E_k=E_k,
        )
    )


def get_privacy_matched_point_relative_directory(
    method,
    target_epsilon,
    E_k,
):
    """Return ``method/epsilon/mu`` for one experiment cell."""
    if method not in PLAN_FILENAMES:
        raise ValueError(f"Unknown plan method {method!r}.")
    epsilon_slug = _path_value_slug(
        target_epsilon,
        "target_epsilon",
    )
    mu_slug = _path_value_slug(E_k, "E_k")
    return Path(
        method,
        f"epsilon_{epsilon_slug}",
        f"mu_{mu_slug}",
    )


def get_privacy_matched_simulations_directory(
    config,
    method,
    target_epsilon,
    E_k,
):
    """Return the simulation root for one privacy-matched cell."""
    exp_config = config.experiment
    return (
        Path(exp_config.output.results_root)
        / str(exp_config.name)
        / str(exp_config.run_id)
        / "simulations"
        / get_privacy_matched_point_relative_directory(
            method=method,
            target_epsilon=target_epsilon,
            E_k=E_k,
        )
    )


def generate_privacy_matched_plan(
    config,
    method,
    target_epsilon,
    E_k,
    plan_filename=None,
):
    """Generate one calibrated Stage-1 plan for an (epsilon, mu) cell."""
    if method not in PLAN_FILENAMES:
        raise ValueError(f"Unknown plan method {method!r}.")
    exp_config = config.experiment
    E_k = _validate_positive_finite(E_k, "E_k")
    calibration = _calibrate_noise_multiplier(
        config=config,
        method=method,
        target_epsilon=target_epsilon,
        expected_num_trials=E_k,
    )
    if method in PAPERNOT_METHODS:
        m = 1
    else:
        m = get_two_stage_settings(exp_config).num_survivors

    if plan_filename is None:
        plan_filename = PLAN_FILENAMES[method][1]
    point_metadata = {
        "target_epsilon": calibration["target_epsilon"],
        "achieved_epsilon": calibration["achieved_epsilon"],
        "delta": calibration["delta"],
        "noise_multiplier": calibration["noise_multiplier"],
        "best_renyi_order": calibration["best_renyi_order"],
        "privacy_calibration": calibration,
    }
    if method in TWO_STAGE_METHODS:
        two_stage_settings = get_two_stage_settings(exp_config)
        point_metadata.update(
            {
                "stage_1_expected_num_trials": E_k,
                "stage_2_expected_num_trials": (
                    two_stage_settings.stage_2_expected_trials
                ),
            }
        )

    return generate_plan(
        exp_config,
        method,
        m,
        [E_k],
        exp_config.num_trials,
        exp_config.run_id,
        exp_config.hp_configuration_ids,
        plan_filename,
        point_metadata=[point_metadata],
        plan_metadata={
            "plan_type": "privacy_matched",
            "selection_signature": get_selection_signature(config),
        },
        plan_directory=get_privacy_matched_plan_directory(
            config=config,
            method=method,
            target_epsilon=target_epsilon,
            E_k=E_k,
        ),
    )


def load_privacy_matched_simulation_plan(config, stage):
    """Load and validate one nested privacy-matched simulation plan."""
    if stage not in {1, 2}:
        raise ValueError(
            f"Simulation stage must be 1 or 2; got {stage!r}."
        )

    exp_config = config.experiment
    method = get_simulation_method(exp_config)
    target_epsilon = _validate_positive_finite(
        exp_config.simulation.target_epsilon,
        "simulation.target_epsilon",
    )
    E_k = _validate_positive_finite(
        exp_config.simulation.mu,
        "simulation.mu",
    )
    plan_filename = PLAN_FILENAMES[method][stage]
    plan_directory = get_privacy_matched_plan_directory(
        config=config,
        method=method,
        target_epsilon=target_epsilon,
        E_k=E_k,
    )
    if stage == 1:
        plan = load_stage_1_plan(
            exp_config,
            plan_filename=plan_filename,
            plan_directory=plan_directory,
        )
    else:
        plan = load_stage_2_plan(
            exp_config,
            plan_filename=plan_filename,
            plan_directory=plan_directory,
        )

    metadata_checks = {
        "plan_type": "privacy_matched",
        "method": method,
        "run_id": str(exp_config.run_id),
        "plan_seed": int(exp_config.seed),
        "hp_configuration_ids": [
            str(hp_id)
            for hp_id in exp_config.hp_configuration_ids
        ],
        "eta": float(exp_config.eta),
        "selection_signature": get_selection_signature(config),
    }
    for key, expected_value in metadata_checks.items():
        if plan.get(key) != expected_value:
            raise ValueError(
                f"Privacy-matched plan {plan_filename} has {key}="
                f"{plan.get(key)!r}, but the experiment configuration "
                f"requires {expected_value!r}. Regenerate the plan."
            )

    expected_sampling_distribution = _sampling_distribution_for_method(
        method
    )
    observed_sampling_distribution = plan.get(
        "sampling_distribution",
        "tnb",
    )
    if observed_sampling_distribution != expected_sampling_distribution:
        raise ValueError(
            f"Privacy-matched plan {plan_filename} has "
            "sampling_distribution="
            f"{observed_sampling_distribution!r}, "
            f"expected {expected_sampling_distribution!r}. Regenerate the "
            "plan."
        )
    if method == "papernot_poisson_baseline":
        if plan.get("selection_method") != "papernot_poisson_top1":
            raise ValueError(
                f"Privacy-matched plan {plan_filename} must use "
                "selection_method='papernot_poisson_top1'."
            )
        expected_fallback = _poisson_k_zero_fallback(exp_config)
        if plan.get("k_zero_fallback") != expected_fallback:
            raise ValueError(
                f"Privacy-matched plan {plan_filename} has a stale or "
                "invalid K=0 fallback definition. Regenerate the plan."
            )
    elif method == "two_stage_poisson_tuning":
        if plan.get("selection_method") != (
            "papernot_poisson_top_m_then_top1"
        ):
            raise ValueError(
                f"Privacy-matched plan {plan_filename} must use the "
                "Poisson two-stage selection method."
            )
        if stage == 2 and plan.get("stage_2_k_zero_fallback") != (
            _poisson_k_zero_fallback(exp_config)
        ):
            raise ValueError(
                f"Privacy-matched plan {plan_filename} has an invalid "
                "Stage-2 K=0 fallback definition. Regenerate the plan."
            )

    expected_top_m = (
        1
        if method in PAPERNOT_METHODS
        else get_two_stage_settings(exp_config).num_survivors
    )
    if plan.get("stage_1_top_m") != expected_top_m:
        raise ValueError(
            f"Privacy-matched plan {plan_filename} has "
            f"stage_1_top_m={plan.get('stage_1_top_m')!r}, expected "
            f"{expected_top_m}. Regenerate the plan."
        )

    points = plan.get("points")
    if not isinstance(points, list) or len(points) != 1:
        raise ValueError(
            f"Privacy-matched plan {plan_filename} must contain "
            "exactly one (target_epsilon, mu) point."
        )
    point = points[0]
    if not isinstance(point, dict):
        raise ValueError(
            f"Privacy-matched plan {plan_filename} point must be "
            "a JSON object."
        )
    trials = point.get("trials")
    configured_num_trials = int(exp_config.num_trials)
    if (
        not isinstance(trials, list)
        or len(trials) != configured_num_trials
    ):
        raise ValueError(
            f"Privacy-matched plan {plan_filename} must contain "
            f"{configured_num_trials} trials. Regenerate the plan."
        )
    if method == "papernot_poisson_baseline":
        require_probability_zero = math.exp(-E_k)
        observed_probability_zero = point.get("probability_K_zero")
        if (
            not isinstance(observed_probability_zero, (int, float))
            or not math.isclose(
                float(observed_probability_zero),
                require_probability_zero,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(
                f"Privacy-matched plan {plan_filename} has an invalid "
                "probability_K_zero. Regenerate the plan."
            )
        for trial_index, trial in enumerate(trials):
            sampled_k = trial.get("sampled_K")
            if (
                isinstance(sampled_k, bool)
                or not isinstance(sampled_k, int)
                or sampled_k < 0
            ):
                raise ValueError(
                    f"Privacy-matched Poisson trial {trial_index} has "
                    f"invalid sampled_K={sampled_k!r}."
                )
            for field in (
                "sampled_hp_configuration_ids",
                "sampled_stage_1_runs",
                "sampled_stage_2_runs",
            ):
                values = trial.get(field)
                if not isinstance(values, list) or len(values) != sampled_k:
                    raise ValueError(
                        f"Privacy-matched Poisson trial {trial_index} has "
                        f"{field} inconsistent with sampled_K={sampled_k}."
                    )

    def require_matching_float(mapping, key, expected, context):
        try:
            observed = float(mapping[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{context} has an invalid {key!r} value."
            ) from error
        if (
            not math.isfinite(observed)
            or not math.isclose(
                observed,
                float(expected),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"{context} has {key}={observed!r}, expected "
                f"{float(expected)!r}. Regenerate the plan."
            )
        return observed

    point_context = f"Privacy-matched plan {plan_filename} point"
    stage_1_E_k_key = (
        "stage_1_E_K"
    if stage == 2 and method in TWO_STAGE_METHODS
        else "E_K"
    )
    require_matching_float(
        point,
        stage_1_E_k_key,
        E_k,
        point_context,
    )
    require_matching_float(
        point,
        "target_epsilon",
        target_epsilon,
        point_context,
    )
    delta = float(exp_config.privacy.delta)
    require_matching_float(point, "delta", delta, point_context)

    noise_multiplier = _validate_positive_finite(
        point.get("noise_multiplier"),
        f"{point_context} noise_multiplier",
    )
    achieved_epsilon = _validate_positive_finite(
        point.get("achieved_epsilon"),
        f"{point_context} achieved_epsilon",
    )
    epsilon_tolerance = max(1e-12, target_epsilon * 1e-9)
    if achieved_epsilon > target_epsilon + epsilon_tolerance:
        raise ValueError(
            f"{point_context} achieved_epsilon={achieved_epsilon!r} "
            f"exceeds target_epsilon={target_epsilon!r}."
        )

    calibration = point.get("privacy_calibration")
    if not isinstance(calibration, dict):
        raise ValueError(
            f"{point_context} must contain a privacy_calibration "
            "object. Regenerate the plan."
        )
    if calibration.get("method") != method:
        raise ValueError(
            f"{point_context} privacy_calibration method does not "
            "match the configured method. Regenerate the plan."
        )
    calibration_context = f"{point_context} privacy_calibration"
    for key, expected_value in (
        ("target_epsilon", target_epsilon),
        ("achieved_epsilon", achieved_epsilon),
        ("delta", delta),
        ("noise_multiplier", noise_multiplier),
    ):
        require_matching_float(
            calibration,
            key,
            expected_value,
            calibration_context,
        )

    if stage == 2 and method in TWO_STAGE_METHODS:
        two_stage_settings = get_two_stage_settings(exp_config)
        require_matching_float(
            point,
            "stage_2_E_K",
            two_stage_settings.stage_2_expected_trials,
            point_context,
        )
        if plan.get("stage_2_selection_m") != 1:
            raise ValueError(
                f"Privacy-matched plan {plan_filename} must have "
                "stage_2_selection_m=1. Regenerate the plan."
            )

    if method == "two_stage_poisson_tuning":
        two_stage_settings = get_two_stage_settings(exp_config)
        stage_1_distribution = PoissonDistribution.from_conditional_mean(
            m=two_stage_settings.num_survivors,
            target_mean=E_k,
        )
        rate_key = (
            "poisson_rate"
            if stage == 1
            else "stage_1_poisson_rate"
        )
        require_matching_float(
            point,
            rate_key,
            stage_1_distribution.mu,
            point_context,
        )
        minimum_k = (
            two_stage_settings.num_survivors if stage == 1 else 0
        )
        for trial_index, trial in enumerate(trials):
            sampled_trial = trial if stage == 1 else trial["trial_stage_2"]
            sampled_k = sampled_trial.get("sampled_K")
            if (
                isinstance(sampled_k, bool)
                or not isinstance(sampled_k, int)
                or sampled_k < minimum_k
            ):
                raise ValueError(
                    f"Privacy-matched Poisson two-stage trial "
                    f"{trial_index} has invalid sampled_K={sampled_k!r}."
                )
            run_field = (
                "sampled_stage_1_runs"
                if stage == 1
                else "sampled_stage_2_runs"
            )
            run_entries = sampled_trial.get(run_field)
            if (
                not isinstance(run_entries, list)
                or len(run_entries) != sampled_k
            ):
                raise ValueError(
                    f"Privacy-matched Poisson two-stage trial "
                    f"{trial_index} has {run_field} inconsistent with "
                    "sampled_K."
                )
        if stage == 2:
            stage_2_expected_k = (
                two_stage_settings.stage_2_expected_trials
            )
            require_matching_float(
                point,
                "stage_2_poisson_rate",
                stage_2_expected_k,
                point_context,
            )
            require_matching_float(
                point,
                "stage_2_probability_K_zero",
                math.exp(-stage_2_expected_k),
                point_context,
            )

    execution_summary = plan.get("execution_summary")
    required_specs_key = f"required_stage_{stage}_run_specs"
    required_specs = (
        execution_summary.get(required_specs_key)
        if isinstance(execution_summary, dict)
        else None
    )
    if not isinstance(required_specs, list):
        raise ValueError(
            f"Privacy-matched plan {plan_filename} does not contain "
            f"a {required_specs_key} list. Regenerate the plan."
        )

    return plan
