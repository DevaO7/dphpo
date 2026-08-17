"""Deterministic TNB trial-plan generation shared by all trainers."""

from collections import Counter
import copy
import json
import math
from pathlib import Path

import numpy as np

from privacy_accounting import dpsgd, rdp_utils, selection_accounting
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
    "two_stage_tuning": {
        1: "two_stage_tuning_stage_1.JSON",
        2: "two_stage_tuning_stage_2.JSON",
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
    planners to attach immutable metadata while retaining the shared TNB
    sampling, seed derivation, and run-deduplication logic.
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

    for point_index, E_K in enumerate(E_K_values):
        gamma = _solve_gamma_for_conditional_mean(
            eta=config.eta,
            m=m,
            target_mean=E_K,
        )
        tnb = TNBDistribution(config.eta, gamma)
        trials = []

        for trial in range(num_trials):
            rng = np.random.default_rng(
                seed=trial + config.seed + point_index
            )
            num_runs = int(tnb.sample_conditional(m, rng))
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
            "gamma": float(gamma),
            "trials": trials,
        }
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
    include_stage_2_specs = method == "papernot_baseline"
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
        "selection_method": "papernot_top1",
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

    if method == "two_stage_tuning":
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
        raise ValueError(
            "The simulation plan has no required Stage-1 runs for "
            f"hyperparameter configuration {hp_configuration_id!r}."
        )
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
        raise ValueError(
            "The Stage-2 plan has no required runs for "
            f"hyperparameter configuration {hp_configuration_id!r}."
        )
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

        stage_2_point = {
            "stage_1_E_K": float(stage_1_point["E_K"]),
            "stage_1_gamma": float(stage_1_point["gamma"]),
            "stage_2_E_K": stage_2_expected_k,
            "stage_2_gamma": float(stage_2_gamma),
            "trials": stage_2_trials,
        }
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
    if "plan_type" in stage_1_plan:
        stage_2_plan["plan_type"] = stage_1_plan["plan_type"]
    return stage_2_plan

def generate_stage_2_plan(
    stage_1_plan,
    config,
    plan_filename="two_stage_tuning_stage_2.JSON",
    *,
    plan_directory=None,
):
    stage_2_plan = build_stage_2_plan(
        stage_1_plan,
        config,
    )
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


def _epsilon_for_noise_multiplier(
    config,
    method,
    expected_num_trials,
    noise_multiplier,
    settings,
):
    """Evaluate end-to-end HPO epsilon for one candidate sigma."""
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
    if method == "papernot_baseline":
        selection_result = selection_accounting.compute_top1_rdp(
            base_rdp_curve=rdp_utils.compose_rdp_curves(
                stage_1_curve,
                stage_2_curve,
            ),
            expected_num_trials=expected_num_trials,
            eta=eta,
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
    else:
        raise ValueError(f"Unknown privacy-calibration method {method!r}.")

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

    return {
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


def _path_value_slug(value, name):
    value = _validate_positive_finite(value, name)
    text = np.format_float_positional(value, trim="-")
    return text.replace(".", "p")


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
    if method == "papernot_baseline":
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
    if method == "two_stage_tuning":
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
        plan_metadata={"plan_type": "privacy_matched"},
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
    }
    for key, expected_value in metadata_checks.items():
        if plan.get(key) != expected_value:
            raise ValueError(
                f"Privacy-matched plan {plan_filename} has {key}="
                f"{plan.get(key)!r}, but the experiment configuration "
                f"requires {expected_value!r}. Regenerate the plan."
            )

    expected_top_m = (
        1
        if method == "papernot_baseline"
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
        if stage == 2 and method == "two_stage_tuning"
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

    if stage == 2 and method == "two_stage_tuning":
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
