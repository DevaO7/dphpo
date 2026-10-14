"""CPU-only Hydra runner for the exp7 top-k accounting comparison."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import math
from pathlib import Path
import subprocess
import time

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from privacy_accounting.dpsgd import compute_dpsgd_rdp
from privacy_accounting.rdp_utils import scale_rdp_curve
from privacy_accounting.tct import (
    TCTOptimizerConfig,
    open_probability,
    positive_integer,
)
from privacy_accounting.topk_comparison import (
    CoverageCalibrationError,
    MethodEvaluation,
    account_dp_composition,
    account_one_shot,
    account_poisson_topk,
    best_accepted_bound,
    calibrate_poisson_coverage,
    calibrate_poisson_expected_compute,
)
from topk_accounting_results import (
    SUMMARY_FIELDS,
    build_summary_row,
    plot_comparison,
    write_csv,
    write_json,
)


_METHODS = (
    "tct_one_shot",
    "basic_dp_composition_one_shot",
    "advanced_dp_composition_one_shot",
    "full_composition_one_shot",
    "poisson_conditioned_topk",
    "poisson_expected_compute_matched_topk",
)


def _unique_values(values, validate, name: str) -> tuple:
    result = tuple(validate(v, name) for v in values)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be nonempty and contain no duplicates.")
    return result


def _positive_float(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def _validate_config(config: DictConfig) -> dict:
    exp = config.experiment
    if (
        exp.runner != "compare_topk_accountants"
        or exp.comparison.mode != "fixed_base_setup"
    ):
        raise ValueError(
            "This runner supports compare_topk_accountants / fixed_base_setup only."
        )
    if tuple(exp.comparison.methods) != _METHODS:
        raise ValueError(
            f"comparison.methods must be {list(_METHODS)}; best bounds are derived."
        )
    if exp.candidates.randomized_sampling != "uniform_with_replacement":
        raise ValueError(
            "The conditioned-Poisson theorem requires the declared with-replacement mechanism."
        )
    if exp.coverage.criterion != "all_configurations":
        raise ValueError("exp7 currently supports all_configurations coverage only.")
    if (
        exp.base_mechanism.type != "dpsgd"
        or exp.base_mechanism.score_privacy != "public_evaluation"
    ):
        raise ValueError("exp7 currently requires DP-SGD with public score evaluation.")
    counts = exp.candidates.num_configurations
    ns = _unique_values(
        counts if isinstance(counts, (list, tuple, ListConfig)) else [counts],
        positive_integer,
        "num_configurations",
    )
    ks = _unique_values(exp.candidates.top_k_values, positive_integer, "top_k_values")
    if max(ks) > min(ns):
        raise ValueError("Every top_k must be <= every num_configurations value.")
    rounds = _unique_values(
        exp.base_mechanism.num_rounds_values, positive_integer, "num_rounds_values"
    )
    sigmas = _unique_values(
        exp.base_mechanism.noise_multipliers, _positive_float, "noise_multipliers"
    )
    targets = _unique_values(
        exp.coverage.target_probabilities, open_probability, "target_probabilities"
    )
    sampling = _positive_float(
        exp.base_mechanism.data_sampling_rate, "data_sampling_rate"
    )
    if sampling > 1:
        raise ValueError("data_sampling_rate must not exceed 1.")
    delta = open_probability(exp.privacy.delta_target, "delta_target")
    minimum = positive_integer(exp.privacy.min_renyi_order, "min_renyi_order")
    maximum = positive_integer(exp.privacy.max_renyi_order, "max_renyi_order")
    if minimum < 2 or maximum <= minimum:
        raise ValueError("Require 2 <= min_renyi_order < max_renyi_order.")
    policy = str(exp.privacy.rdp_conversion)
    if policy not in ("standard", "improved"):
        raise ValueError("rdp_conversion must be standard or improved.")
    reject = exp.privacy.reject_renyi_order_boundary
    if not isinstance(reject, bool):
        raise ValueError("reject_renyi_order_boundary must be a boolean.")
    if exp.tct.base_delta_mode != "optimize":
        raise ValueError("The comparison requires optimized TCT base delta.")
    if (
        tuple(exp.tct.composition_bounds) != ("basic", "advanced")
        or exp.tct.advanced_formula != "lemma_a4"
    ):
        raise ValueError("Require both TCT bounds and advanced_formula=lemma_a4.")
    settings = TCTOptimizerConfig(
        **OmegaConf.to_container(exp.tct.optimizer, resolve=True)
    )
    maximum_trials = _positive_float(
        exp.coverage.maximum_expected_num_trials, "maximum_expected_num_trials"
    )
    if maximum_trials <= max(ks):
        raise ValueError("maximum_expected_num_trials must exceed every top_k.")
    filenames = [str(value) for value in exp.output.values()]
    if len(set(filenames)) != len(filenames) or any(
        Path(s).name != s or s in ("", ".", "..") for s in filenames
    ):
        raise ValueError("Output filenames must be distinct basenames.")
    if not isinstance(exp.plot.enabled, bool):
        raise ValueError("plot.enabled must be a boolean.")
    positive_integer(exp.plot.dpi, "plot.dpi")
    return dict(
        ns=ns,
        ks=ks,
        rounds=rounds,
        sigmas=sigmas,
        targets=targets,
        sampling=sampling,
        delta=delta,
        orders=np.arange(minimum, maximum + 1, dtype=float),
        policy=policy,
        reject=reject,
        optimizer=settings,
        maximum_trials=maximum_trials,
    )


def run_topk_accounting_comparison(
    config: DictConfig, *, output_directory: Path | None = None
) -> dict:
    settings = _validate_config(config)
    exp = config.experiment
    directory = (
        Path(output_directory)
        if output_directory is not None
        else Path(HydraConfig.get().runtime.output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    calibration_cache = {}
    # None is the separate E[K]=N comparison, without a coverage constraint.
    targets = (*settings["targets"], None)
    for n in settings["ns"]:
        for k in settings["ks"]:
            for target in targets:
                try:
                    calibration_cache[n, k, target] = (
                        calibrate_poisson_expected_compute(
                            num_configurations=n, top_k=k
                        )
                        if target is None
                        else calibrate_poisson_coverage(
                            num_configurations=n,
                            top_k=k,
                            target_probability=target,
                            maximum_expected_num_trials=settings["maximum_trials"],
                        )
                    )
                except (CoverageCalibrationError, ArithmeticError) as error:
                    calibration_cache[n, k, target] = str(error)

    rows, diagnostics, base_rows = [], [], []
    one_shot_calls = 0
    dp_composition_calls = 0
    for sigma in settings["sigmas"]:
        # Accounting scales exactly in the number of updates; compute once
        # per sigma and reuse the curve for each T and coverage requirement.
        one_step = compute_dpsgd_rdp(
            {
                "num_rounds": 1,
                "data_sampling_rate": settings["sampling"],
                "sigma_gaussian": sigma,
            },
            settings["orders"],
        )
        for rounds in settings["rounds"]:
            base = scale_rdp_curve(one_step, rounds)
            base_rows.extend(
                dict(
                    num_rounds=rounds,
                    data_sampling_rate=settings["sampling"],
                    noise_multiplier=sigma,
                    renyi_order=float(order),
                    epsilon_rdp=float(epsilon),
                )
                for order, epsilon in zip(base.orders, base.epsilons)
            )
            for n in settings["ns"]:
                ordinary = account_dp_composition(
                    base,
                    num_candidates=n,
                    delta_target=settings["delta"],
                    conversion_policy=settings["policy"],
                    reject_renyi_order_boundary=settings["reject"],
                    optimizer=settings["optimizer"],
                )
                dp_composition_calls += 1
                approx_best = best_accepted_bound(
                    "approx_dp_composition_best", ordinary
                )
                for k in settings["ks"]:
                    tct, full, _ = account_one_shot(
                        base,
                        num_candidates=n,
                        top_k=k,
                        delta_target=settings["delta"],
                        conversion_policy=settings["policy"],
                        reject_renyi_order_boundary=settings["reject"],
                        optimizer=settings["optimizer"],
                    )
                    one_shot_calls += 1
                    best = best_accepted_bound("one_shot_best", (tct, full, *ordinary))
                    fixed = (tct, full, *ordinary, approx_best, best)
                    print(
                        f"Accounted N={n}, T={rounds}, sigma={sigma:g}, k={k}: "
                        f"TCT={tct.epsilon}, best one-shot={best.epsilon}",
                        flush=True,
                    )
                    for target in targets:
                        mode = (
                            "expected_compute_matched"
                            if target is None
                            else "coverage_constrained"
                        )
                        method = (
                            "poisson_expected_compute_matched_topk"
                            if target is None
                            else "poisson_conditioned_topk"
                        )
                        coverage = calibration_cache[n, k, target]
                        if isinstance(coverage, str):
                            poisson = MethodEvaluation(
                                method,
                                None,
                                None,
                                "expected_compute_calibration_failed"
                                if target is None
                                else "coverage_calibration_failed",
                                coverage,
                                None,
                                None,
                                {},
                            )
                        else:
                            try:
                                poisson = account_poisson_topk(
                                    base,
                                    coverage=coverage,
                                    delta_target=settings["delta"],
                                    conversion_policy=settings["policy"],
                                    reject_renyi_order_boundary=settings["reject"],
                                )
                            except ArithmeticError as error:
                                poisson = MethodEvaluation(
                                    method,
                                    None,
                                    None,
                                    "numerical_failure",
                                    str(error),
                                    coverage.expected_num_trials,
                                    coverage.achieved_probability,
                                    {"coverage": asdict(coverage)},
                                )
                        point_id = f"point_{len(diagnostics) + 1:05d}"
                        evaluations = (*fixed, poisson)
                        for evaluation in evaluations:
                            rows.append(
                                build_summary_row(
                                    point_id=point_id,
                                    evaluation=evaluation,
                                    num_configurations=n,
                                    top_k=k,
                                    num_rounds=rounds,
                                    sampling_rate=settings["sampling"],
                                    sigma=sigma,
                                    delta_target=settings["delta"],
                                    conversion_policy=settings["policy"],
                                    target_coverage=target,
                                    workload_mode=mode,
                                    one_shot_best=best.epsilon,
                                    approx_dp_best=approx_best.epsilon,
                                    one_shot_comparison_complete=best.details[
                                        "comparison_complete"
                                    ],
                                    approx_dp_comparison_complete=approx_best.details[
                                        "comparison_complete"
                                    ],
                                )
                            )
                        diagnostics.append(
                            dict(
                                point_id=point_id,
                                num_configurations=n,
                                num_rounds=rounds,
                                sigma=sigma,
                                top_k=k,
                                target_coverage=target,
                                workload_mode=mode,
                                methods=[asdict(item) for item in evaluations],
                            )
                        )

    summary_path = directory / exp.output.summary_filename
    diagnostics_path = directory / exp.output.diagnostics_filename
    base_path = directory / exp.output.base_curves_filename
    write_csv(summary_path, rows, SUMMARY_FIELDS)
    write_json(diagnostics_path, diagnostics)
    write_csv(
        base_path,
        base_rows,
        (
            "num_rounds",
            "data_sampling_rate",
            "noise_multiplier",
            "renyi_order",
            "epsilon_rdp",
        ),
    )
    plots = (
        plot_comparison(rows, directory, dpi=exp.plot.dpi) if exp.plot.enabled else []
    )
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent,
    ).stdout
    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in sorted({row["status"] for row in rows})
    }
    metadata = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete"
        if all(r["status"] == "ok" for r in rows)
        else "complete_with_rejections",
        "point_count": len(diagnostics),
        "row_count": len(rows),
        "status_counts": status_counts,
        "one_shot_accounting_calls": one_shot_calls,
        "dp_composition_accounting_calls": dp_composition_calls,
        "coverage_calibrations": len(settings["ns"])
        * len(settings["ks"])
        * len(settings["targets"]),
        "expected_compute_calibrations": len(settings["ns"]) * len(settings["ks"]),
        "base_one_step_accounting_calls": len(settings["sigmas"]),
        "elapsed_seconds": time.perf_counter() - started,
        "git_head": git_head,
        "git_status": git_status,
        "package_versions": {
            name: version(name)
            for name in ("numpy", "scipy", "dp-accounting", "hydra-core", "matplotlib")
        },
        "resolved_config": OmegaConf.to_container(config, resolve=True),
        "semantics": {
            "base": "Same DP-SGD RDP curve; public score evaluation; record-level Poisson subsampling.",
            "one_shot": "Each of N configurations evaluated once; coverage=1; fixed compute=N*T.",
            "ours": "K~Poisson(mu)|K>=k; configurations sampled uniformly with replacement; expected compute=E[K]*T.",
            "coverage": "Common minimum coverage requirement, not equal realized coverage or utility.",
            "expected_compute_matched": "E[K|K>=k]=N; no coverage constraint; random compute, not a fixed count or mu=N.",
            "ordinary_dp": "Basic N*eps_base; advanced N*eps_base^2/2 + eps_base*sqrt(2*N*log(1/delta_AC)); optimized base delta.",
            "derived_bounds": "approx_dp_composition_best=min(basic DP,advanced DP); one_shot_best=min(TCT,full RDP,basic DP,advanced DP), among accepted bounds.",
            "tct": "Both branches optimize base delta independently; advanced composition follows Lemma A.4.",
            "conditioning": "k=1 is also conditioned; no unconditioned top-1 or zero-trial fallback.",
            "order_rejection": "Applies to selected final, ordinary DP/TCT base, and Poisson auxiliary/source orders.",
            "rejections": "epsilon_bound remains diagnostic; rejected bounds are omitted from epsilon and plots.",
            "ratios": "Suppressed if any component of the corresponding derived comparator is rejected; accepted-only minima remain recorded with comparison_complete=false.",
        },
        "plot_files": [path.name for path in plots],
    }
    metadata_path = directory / exp.output.metadata_filename
    write_json(metadata_path, metadata)
    print(
        f"Saved {len(rows)} rows to {summary_path}; statuses: {status_counts}",
        flush=True,
    )
    return dict(
        summary_path=summary_path,
        diagnostics_path=diagnostics_path,
        base_curves_path=base_path,
        metadata_path=metadata_path,
        plot_paths=plots,
        status_counts=status_counts,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config_topk")
def main(config: DictConfig) -> None:
    run_topk_accounting_comparison(config)


if __name__ == "__main__":
    main()
