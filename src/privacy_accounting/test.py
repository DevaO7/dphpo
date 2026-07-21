import numpy as np
import math
from matplotlib import pyplot as plt
from dpfedavg import compute_dpfedavg_rdp
from rdp_utils import convert_rdp_to_approx_dp, compose_rdp_curves
from selection_accounting import (
    compute_top1_rdp,
    compute_top_m_rdp,
    compute_two_stage_rdp,
)

def plotting():
    orders = np.arange(2, 101)

    low_resource_config = {
        "T": 150,
        "K": 6,
        "M": 100,
        "l": 0.21,
        "s": 0.2,
        "sigma_gaussian": 20,
        "sigma_is_actual": False
    }

    high_resource_config = {
        "T": 150,
        "K": 50,
        "M": 100,
        "l": 0.21,
        "s": 0.2,
        "sigma_gaussian": 20,
        "sigma_is_actual": False
    }

    papernot_high_resource_config = {
        "T": 300,
        "K": 50,
        "M": 100,
        "l": 0.21,
        "s": 0.2,
        "sigma_gaussian": 20,
        "sigma_is_actual": False
    }

    low_resource_curve = compute_dpfedavg_rdp(
        config=low_resource_config,
        orders=orders,
        accounting_method="numerical",
    )

    high_resource_curve = compute_dpfedavg_rdp(
        config=high_resource_config,
        orders=orders,
        accounting_method="numerical",
    )

    papernot_high_resource_curve = compute_dpfedavg_rdp(
        config=papernot_high_resource_config,
        orders=orders,
        accounting_method="numerical",
    )

    papernot_optimized_resource_curve = compose_rdp_curves(
        low_resource_curve, high_resource_curve
    )

    delta = 1e-5

    E_k_min = 3.05
    E_k_max=1_000_000
    num_E_k_points=120
    eta=0

    E_k_values = np.logspace(
            math.log10(E_k_min),
            math.log10(E_k_max),
            num_E_k_points,
        )

    eps_top1_list = []
    eps_topm_list = []
    eps_two_stage_list = []
    eps_top1_optimal_list = []

    for E_k in E_k_values:
        top1_result = compute_top1_rdp(
            base_rdp_curve=papernot_high_resource_curve,
            expected_num_trials=E_k,
            eta=0,
        )

        top1_optimal_result = compute_top1_rdp(
            base_rdp_curve=papernot_optimized_resource_curve,
            expected_num_trials=E_k,
            eta=0,
        )

        topm_result = compute_top_m_rdp(
            base_rdp_curve=low_resource_curve,
            m=3,
            expected_num_trials=E_k,
            eta=0,
        )

        two_stage_result = compute_two_stage_rdp(
            stage_1_base_rdp_curve=low_resource_curve,
            stage_2_base_rdp_curve=high_resource_curve,
            m=3,
            expected_num_trials_stage_1=E_k,
            expected_num_trials_stage_2=3,
            eta_stage_1=0,
            eta_stage_2=0,
        )

        top1_dp = convert_rdp_to_approx_dp(
            top1_result.rdp_curve,
            delta=delta,
        )

        topm_dp = convert_rdp_to_approx_dp(
            topm_result.rdp_curve,
            delta=delta,
        )

        two_stage_dp = convert_rdp_to_approx_dp(
            two_stage_result.rdp_curve,
            delta=delta,
        )

        top1_optimal_dp = convert_rdp_to_approx_dp(
            top1_optimal_result.rdp_curve,
            delta=delta,
        )

        eps_top1_list.append(top1_dp.epsilon)
        eps_top1_optimal_list.append(top1_optimal_dp.epsilon)
        eps_topm_list.append(topm_dp.epsilon)
        eps_two_stage_list.append(two_stage_dp.epsilon)

    plt.figure(figsize=(6, 5))

    plt.plot(
        eps_top1_list,
        E_k_values,
        label=(
            fr"Top-1, $\eta={eta}$, "
        ),
    )

    plt.plot(
        eps_two_stage_list,
        E_k_values,
        label=(
            fr"Two-Stage, $\eta={eta}$, "
        ),
    )

    plt.plot(
        eps_topm_list,
        E_k_values,
        label=(
            fr"Top-{3}, $\eta={eta}$, "
        ),
    )

    plt.plot(
        eps_top1_optimal_list,
        E_k_values,
        label=(
            fr"Top-{1}-Optimal, $\eta={eta}$, "
        ),
    )

    plt.yscale("log")

    plt.xlabel(fr"$\varepsilon$ for $(\varepsilon, {delta:.1e})$-DP")
    plt.ylabel(r"$\mathbb{E}[K]$")

    plt.title(fr"DP-FedAvg with Papernot top-1 vs top-{3} vs two-stage")
    plt.legend()
    plt.tight_layout()

    filename = 'test.png'
    plt.savefig(filename, dpi=300)

def finding_E_K():

    orders = np.arange(2, 101)

    low_resource_config = {
        "T": 150,
        "K": 6,
        "M": 100,
        "l": 0.21,
        "s": 0.2,
        "sigma_gaussian": 20,
        "sigma_is_actual": False
    }

    high_resource_config = {
        "T": 150,
        "K": 50,
        "M": 100,
        "l": 0.21,
        "s": 0.2,
        "sigma_gaussian": 20,
        "sigma_is_actual": False
    }

    low_resource_curve = compute_dpfedavg_rdp(
        config=low_resource_config,
        orders=orders,
        accounting_method="numerical",
    )

    high_resource_curve = compute_dpfedavg_rdp(
        config=high_resource_config,
        orders=orders,
        accounting_method="numerical",
    )

    target_privacy_guarantees = [2, 2.5, 3, 3.5, 4, 4.5, 5]
    E_k_min = 3.05
    E_k_max=1_000_000
    num_E_k_points=120

    E_k_values = np.logspace(
            math.log10(E_k_min),
            math.log10(E_k_max),
            num_E_k_points,
        )

    eps_top1_list = []
    eps_topm_list = []
    eps_two_stage_list = []

    for E_k in E_k_values:
        top1_result = compute_top1_rdp(
            base_rdp_curve=high_resource_curve,
            expected_num_trials=E_k,
            eta=0,
        )

        topm_result = compute_top_m_rdp(
            base_rdp_curve=low_resource_curve,
            m=3,
            expected_num_trials=E_k,
            eta=0,
        )

        two_stage_result = compute_two_stage_rdp(
            stage_1_base_rdp_curve=low_resource_curve,
            stage_2_base_rdp_curve=high_resource_curve,
            m=3,
            expected_num_trials_stage_1=E_k,
            expected_num_trials_stage_2=3,
            eta_stage_1=0,
            eta_stage_2=0,
        )

        delta = 1e-5

        top1_dp = convert_rdp_to_approx_dp(
            top1_result.rdp_curve,
            delta=delta,
        )

        topm_dp = convert_rdp_to_approx_dp(
            topm_result.rdp_curve,
            delta=delta,
        )

        two_stage_dp = convert_rdp_to_approx_dp(
            two_stage_result.rdp_curve,
            delta=delta,
        )

        eps_top1_list.append(top1_dp.epsilon)
        eps_topm_list.append(topm_dp.epsilon)
        eps_two_stage_list.append(two_stage_dp.epsilon)

    for target_privacy in target_privacy_guarantees:
        # Find the E_k value that corresponds to the target privacy guarantee
        closest_index_top1 = np.argmin(np.abs(np.array(eps_top1_list) - target_privacy))
        closest_index_topm = np.argmin(np.abs(np.array(eps_topm_list) - target_privacy))
        closest_index_two_stage = np.argmin(np.abs(np.array(eps_two_stage_list) - target_privacy))

        print(f"Target Privacy: {target_privacy}")
        print(f"Top-1: E[K] ~ {E_k_values[closest_index_top1]:.2f}, Epsilon ~ {eps_top1_list[closest_index_top1]:.2f}")
        print(f"Top-m: E[K] ~ {E_k_values[closest_index_topm]:.2f}, Epsilon ~ {eps_topm_list[closest_index_topm]:.2f}")
        print(f"Two-Stage: E[K] ~ {E_k_values[closest_index_two_stage]:.2f}, Epsilon ~ {eps_two_stage_list[closest_index_two_stage]:.2f}")
        print("-" * 50)

def E_k_compute_plot():
    C1 = 150*6
    C2 = 150*50
    E_k_two_stage = np.arange(3, 50, 1)
    resources_two_stage = E_k_two_stage *(C1) + 3*C2

    matched_E_k_top1_values = []
    for E_k in E_k_two_stage:
        E_k_top1 = (E_k*C1+3*C2)/(C1+C2)
        matched_E_k_top1_values.append(E_k_top1)

    plt.figure(figsize=(6, 5))
    plt.plot(resources_two_stage, matched_E_k_top1_values,  label="E_K for top-1")
    plt.plot(resources_two_stage, E_k_two_stage, label="E_K for Two-Stage")
    plt.xlabel("Compute")
    plt.ylabel("E[K]")
    plt.title("E[K] vs Compute for Top-1 and Two-Stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig("E_K_compute_plot.png", dpi=300)

def matched_search_breadth_plot():
    C1 = 150 * 6
    C2 = 150 * 50
    E_k_stage2 = 3

    E_k_two_stage = np.arange(4, 50)

    E_k_top1_matched = (
        E_k_two_stage * C1
        + E_k_stage2 * C2
    ) / (C1 + C2)

    plt.figure(figsize=(6, 5))

    plt.plot(
        E_k_two_stage,
        E_k_top1_matched,
        label="Compute-matched direct top-1",
    )

    plt.plot(
        E_k_two_stage,
        E_k_two_stage,
        linestyle="--",
        label="Equal search breadth",
    )

    plt.xlabel(r"Two-stage stage-1 $\mathbb{E}[K]$")
    plt.ylabel(r"Compute-matched direct top-1 $\mathbb{E}[K]$")
    plt.title("Search breadth under equal expected compute")
    plt.legend()
    plt.tight_layout()
    plt.savefig("compute_matched_search_breadth.png", dpi=300)


if __name__== "__main__":
    matched_search_breadth_plot()