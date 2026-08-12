"""Deterministic TNB trial-plan generation shared by all trainers."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from privacy_accounting.tnb import (
    TNBDistribution,
    _solve_gamma_for_conditional_mean,
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
):
    """Generate and persist a deterministic static HPO trial plan."""
    if method not in PLAN_FILENAMES:
        raise ValueError(
            f"Unknown plan method {method!r}."
        )

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
    ) = _build_required_stage_run_specs(
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

    plan_directory = (
        Path(config.output.results_root)
        / str(config.name)
        / str(run_id)
        / "plan"
    )
    plan_directory.mkdir(parents=True, exist_ok=True)
    plan_path = plan_directory / plan_filename

    with plan_path.open(mode="w", encoding="utf-8") as file:
        json.dump(plan, file, indent=4, allow_nan=False)

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
