import numpy as np
import matplotlib.pyplot as plt


class Papernot_Baseline:
    def __init__(self, config):
        self.config = config

    def calculate_E_K_given_compute(self, E_K, N):

        return 0
    
    #TODO
    def run_simulation(self, E_K, seed):
        utility = 0
        best_HP = 0
        return best_HP, utility

class N_Stage_Method:
    def __init__(self, config):
        self.config = config
    
    #TODO
    def run_simulation(self, E_K, seed):
        utility = 0
        best_HP = 0
        return best_HP, utility

    def calculate_compute_given_E_K(self, E_K):
        # Placeholder for the actual calculation logic
        # This function should return the compute value based on E_K and other parameters in config
        return E_K * self.config.some_factor  # Example computation




def calculate_compute_N_stage(E_K, config):
    # Placeholder for the actual compute calculation logic
    # This function should return the compute value based on E_K and other parameters in config
    return E_K * config.some_factor  # Example computation


def utility_compute_plot(config):
    baseline = Papernot_Baseline(config)
    n_stage_tuning = N_Stage_Method(config)
    E_K_values_N_stage = np.arange(config.min_E_K, config.max_E_K + 1, config.step_E_K)
    fixed_compute = np.zeros_like(E_K_values_N_stage, dtype=float)
    utility_values_N_stage = np.zeros((len(E_K_values_N_stage),config.num_trials), dtype=float)
    utility_values_papernot_baseline = np.zeros_like(utility_values_N_stage, dtype=float)
    E_K_values_papernot_baseline  = np.zeros_like(E_K_values_N_stage, dtype=float)
    for i, E_K in enumerate(E_K_values_N_stage):
        fixed_compute[i] = n_stage_tuning.calculate_compute_given_E_K(E_K)
        E_K_values_papernot_baseline[i] = baseline.calculate_E_K_given_compute(fixed_compute[i])
    for i, compute in enumerate(fixed_compute):
        for trial in range(config.num_trials):
            best_HP_baseline, utility_values_papernot_baseline[i, trial] = baseline.run_simulation(E_K_values_papernot_baseline[i], seed=trial)
            best_HP, utility_values_N_stage[i, trial] = n_stage_tuning.run_simulation(E_K_values_N_stage[i], seed=trial)

    plt.plot(fixed_compute, utility_values_papernot_baseline.mean(axis=1), label="Papernot Baseline")
    plt.plot(fixed_compute, utility_values_N_stage.mean(axis=1), label="N-Stage Method")
    plt.xlabel("E_K")
    plt.ylabel("Utility")
    plt.title("Utility vs E_K")
    plt.legend()
    plt.savefig(config.plot_filename, dpi=300)

if __name__ == "__main__":
    config = dict() #TODO
    utility_compute_plot(config)