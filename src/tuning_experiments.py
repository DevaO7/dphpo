import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from privacy_accounting.tnb import _solve_gamma_for_conditional_mean, TNBDistribution
from pathlib import Path
import json

class Papernot_Baseline:
    def __init__(self, config):
        self.config = config

    def calculate_E_K_given_compute(self, compute, local_updates_schedule):
        E_K = compute/sum(local_updates_schedule) 
        return E_K

    #TODO
    def run_simulation(self, E_K, seed, num_stages):
        utility = 0
        best_HP = 0
        # runs N stages. And each stage all of the chosen candidates will be running. 
        # So we only sample once. But we continue to use the same model as we progress through. Do we want to execute in parallel or sequence. 
        # I think running them in parallel is absolutely necessary. 
        # So given E_K and number of trials. We will generate K for each trial. And we initalize with 10 HP. Next for each K we will sample K HPs. Now we need to come with a tagging
        
        return best_HP, utility
    
    def generate_plan(self, E_K_values, num_trials, run_id, hp_configuration_ids):
        hp_configuration_counts = {
            hp_id: 0
            for hp_id in hp_configuration_ids
        }
        points = []
        total_num_simulations = 0
        for i, E_K in enumerate(E_K_values):
            gamma = _solve_gamma_for_conditional_mean(eta=self.config.eta, m=1, target_mean=E_K)
            tnb = TNBDistribution(self.config.eta, gamma)
            trials = []
            for trial in range(num_trials):
                rng = np.random.default_rng(seed=trial+run_id+i)
                K = tnb.sample(rng)
                total_num_simulations += K
                sampled_hp_configuration_ids = rng.choice(hp_configuration_ids, size=K, replace=True)
                trials.append({
                    "trial": trial, 
                    "sampled_K": K,
                    "sampled_hp_configuration_ids": sampled_hp_configuration_ids.tolist()
                })
                for hp_id in sampled_hp_configuration_ids:
                    hp_configuration_counts[hp_id] += 1
            points.append({
                "E_K": E_K,
                "gamma": gamma,
                "trials": trials,
            })
        plan = {
            "method": "papernot_top1",
            "eta": self.config.eta,
            "plan_seed": run_id,
            "points": points,
            "execution_summary": {
                "total_num_simulations": (
                    total_num_simulations
                ),
                "hp_configuration_counts": (
                    hp_configuration_counts
                ),
            },
        }
        results_root = Path(
            self.config.output.results_root
        )

        experiment_name = str(
            self.config.name
        )

        plan_directory = (
            results_root
            / experiment_name
            / f'run_id_{run_id}'
            / "plan"
        )

        plan_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        plan_path = (
            plan_directory
            / self.config.output.plan_filename
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

class N_Stage_Method:
    def __init__(self, config):
        self.config = config.n_stage_tuning
    
    #TODO
    def run_simulation(self, E_K, seed):
        utility = 0
        best_HP = 0
        return best_HP, utility

    def calculate_compute_given_E_K(self, E_K, local_updates_schedule):
        E_K_each_stage = [E_K] + self.config.E_K_each_stage
        compute = 0
        for i, E_K in enumerate(E_K_each_stage):
            compute += E_K * local_updates_schedule[i]
        return compute
    
    def generate_plan(self, E_K_values, num_trials, run_id):
        pass


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
        baseline.generate_plan(E_K_values_papernot_baseline, exp_config.num_trials, exp_config.run_id, exp_config.hp_configuration_ids)
        exit()
        n_stage_tuning.generate_plan(E_K_values_N_stage, exp_config.num_trials, exp_config.run_id, exp_config.hp_configuration_ids)
    
    if exp_config.run_mode.run_base_simulation:
        baseline.run_simulation(exp_config.base_E_K, exp_config.run_id, exp_config.num_stages)




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
