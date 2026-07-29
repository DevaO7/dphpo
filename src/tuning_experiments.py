import numpy as np
import matplotlib.pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from privacy_accounting.tnb import _solve_gamma_for_conditional_mean, TNBDistribution
from pathlib import Path
import json
from collections import Counter
from utils.data_utils import get_data_loaders, set_seed
from flearn.trainmodel import models
from flearn.servers.server_avg import FedAvg


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
    m,
    E_K_values,
    num_trials,
    run_id,
    hp_configuration_ids,
    plan_filename,
):
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

    plan = {
        "method": "papernot_top1",
        "eta": config.eta,
        "plan_seed": run_id,
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
        },
    }

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


class Papernot_Baseline:
    def __init__(self, config):
        self.config = config

    def calculate_E_K_given_compute(self, compute, local_updates_schedule):
        E_K = compute/sum(local_updates_schedule)
        return E_K


class N_Stage_Method:
    def __init__(self, config):
        self.config = config.n_stage_tuning
    
    def calculate_compute_given_E_K(self, E_K, local_updates_schedule):
        E_K_each_stage = [E_K] + self.config.E_K_each_stage
        compute = 0
        for i, E_K in enumerate(E_K_each_stage):
            compute += E_K * local_updates_schedule[i]
        return compute
    
def utility_compute_plot(config: DictConfig) -> None:
    exp_config = config.experiment
    baseline = Papernot_Baseline(exp_config)
    n_stage_tuning = N_Stage_Method(exp_config)

    if exp_config.run_mode.generate_plan:
        E_K_values_N_stage = exp_config.base_E_K_list
        local_updates_schedule = exp_config.local_updates_schedule
        fixed_compute = np.zeros_like(E_K_values_N_stage, dtype=float)
        E_K_values_papernot_baseline = np.zeros_like(E_K_values_N_stage, dtype=float)
        for i, E_K in enumerate(E_K_values_N_stage):
            fixed_compute[i] = n_stage_tuning.calculate_compute_given_E_K(E_K, local_updates_schedule)
            E_K_values_papernot_baseline[i] = baseline.calculate_E_K_given_compute(fixed_compute[i], local_updates_schedule)
        generate_plan(exp_config, 1, E_K_values_papernot_baseline, exp_config.num_trials, exp_config.run_id, exp_config.hp_configuration_ids, "papernot_baseline.JSON")
        generate_plan(exp_config, exp_config.n_stage_tuning.E_K_each_stage[0], E_K_values_N_stage, exp_config.num_trials, exp_config.run_id, exp_config.hp_configuration_ids, "two_stage_tuning_stage_1.JSON")


    if exp_config.run_mode.run_baseline_simulation:
        set_seed(config.run_settings.seed)
        if config.dataset.name == 'synthetic':
            model = getattr(models, config.dataset.model_name)(input_dim=config.dataset.dim_input, output_dim=config.dataset.dim_output)
        else:
            model = getattr(models, config.dataset.model_name)()
        learning_rate = get_selected_learning_rate(config)
        if config.server.constant_global_step == 'Fixed':
            global_step = config.server.global_step
            local_step = learning_rate
        elif config.server.constant_global_step == 'Adaptive':
            global_step = (config.server.client_ratio*config.dataset.nb_users)**0.5
            local_step = learning_rate/(config.server.local_updates*global_step)
        main_path = Path(HydraConfig.get().runtime.output_dir)/ "simulations"
        clipping_value = config.server.max_grad_norm
        for seed in range(exp_config.simulation.times):
            save_path = main_path / config.experiment.simulation.run_hp_configuration / f"seed_{seed}"
            save_path.mkdir(parents=True, exist_ok=True)
            train_data_loader, test_data_loader = get_data_loaders(config, per_client_loader=True)
            server = FedAvg(
                    model=model,
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
                    max_grad_norm=clipping_value, 
                    x_label=config.dataset.x_label,
                    y_label=config.dataset.y_label, 
                    client_sampling_scheme=config.server.client_sampling_scheme, 
                    data_sampling_scheme=config.server.data_sampling_scheme,
                    stage=exp_config.simulation.stage, 
                    stage_1_end=exp_config.simulation.stage_1_end, 
                    base_seed=seed
                )
            server.train()
            OmegaConf.save(config, save_path / f"stage_{exp_config.simulation.stage}_config.yaml", resolve=True)

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
