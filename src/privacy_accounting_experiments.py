import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from privacy_accounting.rdp_utils import (
    RdpCurve,
    compose_rdp_curves,
    convert_rdp_to_approx_dp,
)
from privacy_accounting.selection_accounting import (
    compute_top1_rdp,
    compute_top_m_rdp,
)


def compare_with_peeling_plot():
    """Compare direct top-m accounting with composed top-1 accounting.

    The composed top-1 curve is an accounting baseline. It does not by
    itself establish equivalence to sequentially removing winners from a
    single shared random pool of candidates.
    """
    orders = np.arange(2, 101)
    E_k_min = 3.05
    E_k_max = 1_000_000
    num_E_k_points = 120
    eta = 0.0
    m = 3
    delta = 1e-5

    # base mechanism
    base_mechanism = RdpCurve(
        orders=orders,
        epsilons=0.1 * orders,
    )

    E_k_values = np.logspace(
        math.log10(E_k_min),
        math.log10(E_k_max),
        num_E_k_points,
    )

    eps_topm_list = []
    eps_peeling_list = []
    for E_k in E_k_values:
        # papernot top-1
        peeling_rdp_curves = []
        for peel in range(m):
            papernot_top1_result = compute_top1_rdp(
                base_rdp_curve=base_mechanism,
                expected_num_trials=E_k - peel,
                eta=eta,
            )
            peeling_rdp_curves.append(
                papernot_top1_result.rdp_curve
            )

        peeling_composed = compose_rdp_curves(*peeling_rdp_curves)

        # top-m
        top_m_result = compute_top_m_rdp(
            base_rdp_curve=base_mechanism,
            m=m,
            expected_num_trials=E_k,
            eta=eta,
        )
        peeling_composed_dp = convert_rdp_to_approx_dp(
            peeling_composed,
            delta=delta,
        )
        top_m_dp = convert_rdp_to_approx_dp(
            top_m_result.rdp_curve,
            delta=delta,
        )
        eps_peeling_list.append(peeling_composed_dp.epsilon)
        eps_topm_list.append(top_m_dp.epsilon)

    plt.figure(figsize=(10, 6))
    plt.plot(
        E_k_values,
        eps_peeling_list,
        label="Composed top-1 baseline",
        marker="o",
    )
    plt.plot(
        E_k_values,
        eps_topm_list,
        label="Direct top-m",
        marker="x",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Expected number of trials $\mathbb{E}[K]$")
    plt.ylabel("Epsilon (log scale)")
    plt.title("Privacy Accounting: Peeling vs Top-M")
    plt.legend()
    plt.tight_layout()

    output_path = Path(
        "results/exp0_privacy_accounting/"
        "privacy_accounting_comparison.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


if __name__ == "__main__":
    compare_with_peeling_plot()
