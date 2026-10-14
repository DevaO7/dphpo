"""Persistence and plots for deterministic single-stage accounting points."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


SUMMARY_FIELDS = (
    "point_id",
    "method",
    "status",
    "diagnostic",
    "num_configurations",
    "top_k",
    "num_rounds",
    "data_sampling_rate",
    "noise_multiplier",
    "delta_target",
    "rdp_conversion",
    "workload_mode",
    "target_coverage",
    "achieved_coverage",
    "poisson_rate",
    "conditioning_probability",
    "expected_num_trials",
    "compute_type",
    "expected_compute",
    "epsilon",
    "epsilon_bound",
    "accountant_best",
    "eps_base",
    "delta_base",
    "candidate_delta_sum",
    "q_tct",
    "alpha_tct",
    "effective_accesses",
    "delta_tail_bound",
    "delta_ac",
    "best_renyi_order",
    "auxiliary_order",
    "epsilon_ratio_to_one_shot_best",
    "epsilon_ratio_to_approx_dp_best",
    "one_shot_comparison_complete",
    "approx_dp_comparison_complete",
)


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def plot_comparison(rows: list[dict], directory: Path, *, dpi: int = 150) -> list[Path]:
    """Separate coverage targets and expected-compute matching in each figure."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    labels = {
        "tct_one_shot": "TCT (raw bound)",
        "basic_dp_composition_one_shot": "Basic DP composition",
        "advanced_dp_composition_one_shot": "Advanced DP composition",
        "full_composition_one_shot": "Full RDP composition",
        "approx_dp_composition_best": "Best accepted ordinary DP",
        "one_shot_best": "Best accepted one-shot",
        "poisson_conditioned_topk": "Ours: coverage constrained",
        "poisson_expected_compute_matched_topk": "Ours: E[K]=N",
    }
    colors = dict(zip(labels, ("C1", "C4", "C5", "C2", "C6", "C3", "C0", "C0")))
    paths = []
    setups = sorted({(r["num_rounds"], r["noise_multiplier"]) for r in rows})
    targets = sorted(
        {r["target_coverage"] for r in rows if r["target_coverage"] is not None}
    )
    if any(r["target_coverage"] is None for r in rows):
        targets.append(None)
    for rounds, sigma in setups:
        selected = [
            r
            for r in rows
            if (r["num_rounds"], r["noise_multiplier"]) == (rounds, sigma)
        ]
        counts = sorted({r["num_configurations"] for r in selected})
        tasks = [
            (kind, "num_configurations", n, "top_k")
            for n in counts
            for kind in ("epsilon_vs_top_k", "privacy_ratio", "compute_vs_privacy")
        ]
        if len(counts) > 1:
            tasks.extend(
                ("epsilon_vs_num_candidates", "top_k", k, "num_configurations")
                for k in sorted({r["top_k"] for r in selected})
            )
        for kind, fixed_field, fixed_value, x_field in tasks:
            figure, axes = plt.subplots(
                1, len(targets), figsize=(5 * len(targets), 4), squeeze=False
            )
            for axis, target in zip(axes[0], targets):
                panel = [
                    r
                    for r in selected
                    if r["target_coverage"] == target and r[fixed_field] == fixed_value
                ]
                if kind == "privacy_ratio":
                    ours = (
                        "poisson_expected_compute_matched_topk"
                        if target is None
                        else "poisson_conditioned_topk"
                    )
                    for method, field, label in (
                        (
                            ours,
                            "epsilon_ratio_to_one_shot_best",
                            "Ours / best one-shot",
                        ),
                        (
                            "tct_one_shot",
                            "epsilon_ratio_to_approx_dp_best",
                            "TCT / best ordinary DP",
                        ),
                    ):
                        values = sorted(
                            (
                                r
                                for r in panel
                                if r["method"] == method and r[field] is not None
                            ),
                            key=lambda r: r[x_field],
                        )
                        axis.plot(
                            [r[x_field] for r in values],
                            [r[field] for r in values],
                            marker="o",
                            label=label,
                            color=colors[method],
                        )
                    axis.axhline(1.0, linestyle="--", color="gray")
                    axis.set(xlabel="Released outputs k", ylabel="Epsilon ratio")
                else:
                    for method, label in labels.items():
                        values = sorted(
                            (
                                r
                                for r in panel
                                if r["method"] == method and r["epsilon"] is not None
                            ),
                            key=lambda r: r[x_field],
                        )
                        if not values:
                            continue
                        if kind != "compute_vs_privacy":
                            xs = [r[x_field] for r in values]
                            ys = [r["epsilon"] for r in values]
                            axis.set(
                                xlabel="Candidates N"
                                if x_field == "num_configurations"
                                else "Released outputs k",
                                ylabel="Selection epsilon",
                            )
                        else:
                            xs = [r["expected_compute"] for r in values]
                            ys = [r["epsilon"] for r in values]
                            axis.set(
                                xlabel="Expected optimizer updates",
                                ylabel="Selection epsilon",
                            )
                            for x, y, row in zip(xs, ys, values):
                                axis.annotate(f"k={row['top_k']}", (x, y), fontsize=7)
                        axis.plot(
                            xs,
                            ys,
                            marker="o",
                            label=label,
                            color=colors[method],
                            linestyle="--" if method.endswith("_best") else "-",
                        )
                rejected = sum(r["status"] != "ok" for r in panel)
                axis.set_title(
                    (
                        "Matched expected compute E[K]=N\nCoverage unconstrained"
                        if target is None
                        else f"Coverage requirement {target:.0%}\nOne-shot coverage: 100%"
                    )
                )
                if kind != "privacy_ratio" and any(
                    r["epsilon"] is not None and r["epsilon"] > 0 for r in panel
                ):
                    axis.set_yscale("symlog", linthresh=1e-2)
                if x_field == "num_configurations":
                    axis.set_xscale("log")
                if rejected:
                    axis.text(
                        0.02,
                        0.02,
                        f"{rejected} rejected rows; see CSV",
                        transform=axis.transAxes,
                        fontsize=8,
                    )
                axis.grid(alpha=0.25)
                handles, legend_labels = axis.get_legend_handles_labels()
                if handles:
                    axis.legend(handles, legend_labels, fontsize=8)
            figure.suptitle(
                f"{'N' if fixed_field == 'num_configurations' else 'k'}={fixed_value}, "
                f"T={rounds}, sampling={selected[0]['data_sampling_rate']:g}, sigma={sigma:g}"
            )
            figure.tight_layout()
            slug = format(sigma, ".12g").replace(".", "p")
            fixed_slug = (
                f"N{fixed_value}"
                if fixed_field == "num_configurations"
                else f"k{fixed_value}"
            )
            path = directory / f"{kind}_{fixed_slug}_T{rounds}_sigma{slug}.png"
            figure.savefig(path, dpi=dpi)
            plt.close(figure)
            paths.append(path)
    return paths


def build_summary_row(
    *,
    point_id: str,
    evaluation,
    num_configurations: int,
    top_k: int,
    num_rounds: int,
    sampling_rate: float,
    sigma: float,
    delta_target: float,
    conversion_policy: str,
    target_coverage: float | None,
    one_shot_best: float | None,
    workload_mode: str = "coverage_constrained",
    approx_dp_best: float | None = None,
    one_shot_comparison_complete: bool = True,
    approx_dp_comparison_complete: bool = True,
) -> dict:
    details = evaluation.details
    row = dict.fromkeys(SUMMARY_FIELDS)
    row.update(
        point_id=point_id,
        method=evaluation.method,
        status=evaluation.status,
        diagnostic=evaluation.diagnostic,
        num_configurations=num_configurations,
        top_k=top_k,
        num_rounds=num_rounds,
        data_sampling_rate=sampling_rate,
        noise_multiplier=sigma,
        delta_target=delta_target,
        rdp_conversion=conversion_policy,
        workload_mode=workload_mode,
        target_coverage=target_coverage,
        achieved_coverage=evaluation.coverage_probability,
        expected_num_trials=evaluation.expected_num_trials,
        compute_type="random_expected"
        if evaluation.method.startswith("poisson_")
        else "fixed",
        expected_compute=(
            evaluation.expected_num_trials * num_rounds
            if evaluation.expected_num_trials is not None
            else None
        ),
        epsilon=evaluation.epsilon,
        epsilon_bound=evaluation.epsilon_bound,
        accountant_best=details.get("accountant_best", details.get("method_best")),
        one_shot_comparison_complete=one_shot_comparison_complete,
        approx_dp_comparison_complete=approx_dp_comparison_complete,
    )
    if "coverage" in details:
        row["poisson_rate"] = details["coverage"]["poisson_rate"]
        row["conditioning_probability"] = details["coverage"][
            "conditioning_probability"
        ]
        row["auxiliary_order"] = details.get("auxiliary_order")
    if "conversion" in details:
        row["best_renyi_order"] = details["conversion"]["best_order"]
    if "base_dp" in details:
        row.update(
            eps_base=details["base_dp"]["epsilon"],
            delta_base=details["base_dp"]["delta"],
            candidate_delta_sum=details["candidate_delta_sum"],
            delta_ac=details["delta_ac"],
            best_renyi_order=details["base_dp"]["best_order"],
        )
    if evaluation.method == "tct_one_shot" and "method_best" in details:
        branch = details[details["method_best"]]
        row.update(
            eps_base=branch["base_dp"]["epsilon"],
            delta_base=branch["base_dp"]["delta"],
            candidate_delta_sum=branch["candidate_delta_sum"],
            q_tct=branch["q"],
            alpha_tct=branch["bound"]["alpha"],
            effective_accesses=branch["bound"]["effective_accesses"],
            delta_tail_bound=branch["bound"]["delta_tail_bound"],
            delta_ac=branch["bound"]["delta_ac"],
            best_renyi_order=branch["base_dp"]["best_order"],
        )
    if evaluation.method.startswith("poisson_") and evaluation.epsilon is not None:
        if (
            one_shot_comparison_complete
            and one_shot_best is not None
            and one_shot_best > 0
        ):
            row["epsilon_ratio_to_one_shot_best"] = evaluation.epsilon / one_shot_best
    if evaluation.method == "tct_one_shot" and evaluation.epsilon is not None:
        if (
            approx_dp_comparison_complete
            and approx_dp_best is not None
            and approx_dp_best > 0
        ):
            row["epsilon_ratio_to_approx_dp_best"] = evaluation.epsilon / approx_dp_best
    for name, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ArithmeticError(f"Nonfinite summary value {name}.")
    return row
