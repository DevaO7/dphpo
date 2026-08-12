import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf


def save_run_spec(save_path, stage, run_spec):
    """Persist a run spec and reject reuse with conflicting metadata."""
    if stage not in {1, 2}:
        raise ValueError(
            f"Simulation stage must be 1 or 2; got {stage!r}."
        )
    if not isinstance(run_spec, dict):
        raise ValueError(
            "run_spec must be a dictionary; "
            f"got {type(run_spec).__name__}."
        )

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
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

    return spec_path


def validate_simulation_stage(config, stage):
    """Validate the configured stage and its cumulative round endpoint."""
    if stage not in {1, 2}:
        raise ValueError(
            f"Simulation stage must be 1 or 2; got {stage!r}."
        )

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
    stage_2_end = int(exp_config.simulation.stage_2_end)

    if stage_1_end <= 0:
        raise ValueError(
            "simulation.stage_1_end must be positive; got "
            f"{stage_1_end!r}."
        )
    if stage_2_end <= stage_1_end:
        raise ValueError(
            "simulation.stage_2_end must be greater than "
            "simulation.stage_1_end; got "
            f"{stage_2_end!r} and {stage_1_end!r}."
        )

    expected_rounds = (
        stage_1_end
        if stage == 1
        else stage_2_end
    )
    if rounds != expected_rounds:
        raise ValueError(
            f"Stage-{stage} simulation requires "
            f"run_settings.rounds={expected_rounds}; "
            f"got {rounds!r}."
        )

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
