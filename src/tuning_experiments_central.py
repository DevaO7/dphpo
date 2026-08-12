import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from utils.seed_utils import set_global_seed
from hpo.execution import (
    get_selected_learning_rate,
    save_run_spec,
    validate_simulation_stage,
)
from hpo.planning import (
    PLAN_FILENAMES,
    generate_plan,
    get_required_simulation_run_specs,
    load_simulation_plan,
)
from utils.hpo_config import get_two_stage_settings
from central.data import get_data_loaders


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
    step_size = get_selected_learning_rate(config)
    simulations_root = (
        Path(HydraConfig.get().runtime.output_dir)
        / "simulations"
    )

    for run_spec in required_run_specs:
        base_seed = int(
            run_spec[f"stage_{stage}_base_seed"]
        )
        set_global_seed(base_seed)
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
            seed=base_seed,
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

    stage_iters = [
        stage_1_end,
        stage_2_end - stage_1_end,
    ]
    return stage_iters


def calculate_compute_given_E_K_two_stage_tuning(
    config,
    E_K,
    stage_compute_schedule,
):
    two_stage_settings = get_two_stage_settings(config)
    E_K_each_stage = [
        E_K,
        two_stage_settings.stage_2_expected_trials,
    ]
    compute = 0
    for i, E_K in enumerate(E_K_each_stage):
        compute += E_K * stage_compute_schedule[i]
    return compute


def calculate_E_K_given_compute_for_papernot(compute, stage_compute_schedule):
    E_K = compute/sum(stage_compute_schedule)
    return E_K


def utility_compute_plot(config: DictConfig) -> None:
    exp_config = config.experiment
    two_stage_settings = get_two_stage_settings(exp_config)

    if exp_config.run_mode.generate_stage_1_plan:
        E_K_values_N_stage = exp_config.base_E_K_list
        stage_compute_schedule = get_stage_compute_schedule(
            exp_config
        )
        fixed_compute = np.zeros_like(E_K_values_N_stage, dtype=float)
        E_K_values_papernot_baseline = np.zeros_like(
            E_K_values_N_stage,
            dtype=float,
        )
        for i, E_K in enumerate(E_K_values_N_stage):
            fixed_compute[i] = (
                calculate_compute_given_E_K_two_stage_tuning(
                    exp_config,
                    E_K,
                    stage_compute_schedule,
                )
            )
            E_K_values_papernot_baseline[i] = (
                calculate_E_K_given_compute_for_papernot(
                    fixed_compute[i],
                    stage_compute_schedule,
                )
            )
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
            two_stage_settings.num_survivors,
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



EXPERIMENT_RUNNERS = {
    "utility_compute_plot": utility_compute_plot,
}


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config_cl",
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
