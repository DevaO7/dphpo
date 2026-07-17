import numpy as np
import math
from matplotlib import pyplot as plt
from dpfedavg import compute_dpfedavg_rdp
from rdp_utils import convert_rdp_to_approx_dp
from selection_accounting import (
    compute_top1_rdp,
    compute_top_m_rdp,
    compute_two_stage_rdp,
)


orders = np.arange(2, 101)

low_resource_config = {
    "T": 150,
    "K": 6,
    "M": 100,
    "l": 0.21,
    "s": 0.2,
    "sigma_gaussian": 20,
}

high_resource_config = {
    "T": 150,
    "K": 50,
    "M": 100,
    "l": 0.21,
    "s": 0.2,
    "sigma_gaussian": 20,
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


top1_result = compute_top1_rdp(
    base_rdp_curve=high_resource_curve,
    expected_num_trials=10,
    eta=0,
)

topm_result = compute_top_m_rdp(
    base_rdp_curve=low_resource_curve,
    m=3,
    expected_num_trials=10,
    eta=0,
)

two_stage_result = compute_two_stage_rdp(
    stage_1_base_rdp_curve=low_resource_curve,
    stage_2_base_rdp_curve=high_resource_curve,
    m=3,
    expected_num_trials_stage_1=10,
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

plt.yscale("log")

plt.xlabel(fr"$\varepsilon$ for $(\varepsilon, {delta:.1e})$-DP")
plt.ylabel(r"$\mathbb{E}[K]$")

plt.title(fr"DP-FedAvg with Papernot top-1 vs top-{3} vs two-stage")
plt.legend()
plt.tight_layout()

filename = 'test.png'
plt.savefig(filename, dpi=300)
