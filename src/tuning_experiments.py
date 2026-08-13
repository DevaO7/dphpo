import numpy as np
import matplotlib.pyplot as plt
import hydra
import csv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json
from utils.data_utils import get_data_loaders
from utils.seed_utils import set_global_seed
from flearn.trainmodel import models
from flearn.servers.server_avg import FedAvg
from privacy_accounting.dpfedavg import compute_dpfedavg_rdp
from privacy_accounting.rdp_utils import convert_rdp_to_approx_dp, compose_rdp_curves
from privacy_accounting.selection_accounting import (
    compute_top1_rdp,
    compute_two_stage_rdp,
)
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
from hpo.results import (
    RESULT_FILENAMES,
    build_privacy_compute_points,
    get_compilation_paths,
    get_evaluation_settings,
    load_compilation_plan,
    compile_papernot_results,
    compile_two_stage_results,
    generate_stage_2_plan_from_results,
)
from utils.hpo_config import get_two_stage_settings

def build_model(config):
    model_class = getattr(
        models,
        config.dataset.model_name,
    )
    if config.dataset.name == "synthetic":
        return model_class(
            input_dim=config.dataset.dim_input,
            output_dim=config.dataset.dim_output,
        )
    return model_class()


def get_learning_steps(config):
    learning_rate = get_selected_learning_rate(config)
    step_mode = str(config.server.constant_global_step)
    if step_mode == "Fixed":
        return learning_rate, config.server.global_step
    if step_mode == "Adaptive":
        global_step = (
            config.server.client_ratio
            * config.dataset.nb_users
        ) ** 0.5
        local_step = learning_rate / (
            config.server.local_updates * global_step
        )
        return local_step, global_step
    raise ValueError(
        "server.constant_global_step must be 'Fixed' or "
        f"'Adaptive'; got {step_mode!r}."
    )


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
    local_step, global_step = get_learning_steps(config)
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
            per_client_loader=True,
        )
        server = FedAvg(
            model=build_model(config),
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
            max_grad_norm=config.server.max_grad_norm,
            x_label=config.dataset.x_label,
            y_label=config.dataset.y_label,
            client_sampling_scheme=(
                config.server.client_sampling_scheme
            ),
            data_sampling_scheme=(
                config.server.data_sampling_scheme
            ),
            stage=stage,
            stage_1_end=exp_config.simulation.stage_1_end,
            base_seed=base_seed,
            stage_1_source_path=stage_1_source_path,
        )
        server.train()
        OmegaConf.save(
            config,
            save_path / f"stage_{stage}_config.yaml",
            resolve=True,
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

    local_updates_schedule = [
        int(local_updates)
        for local_updates in config.local_updates_schedule
    ]
    if (
        len(local_updates_schedule) != 2
        or any(
            local_updates <= 0
            for local_updates in local_updates_schedule
        )
    ):
        raise ValueError(
            "local_updates_schedule must contain two positive "
            "integers, one for each stage."
        )

    stage_rounds = [
        stage_1_end,
        stage_2_end - stage_1_end,
    ]
    return [
        rounds * local_updates
        for rounds, local_updates in zip(
            stage_rounds,
            local_updates_schedule,
        )
    ]

def build_trial_result_rows(compiled_results):
    rows = []
    for method, result in compiled_results.items():
        utility_metrics = result["evaluation"]["utility"][
            "metrics"
        ]
        for point_index, point in enumerate(result["points"]):
            for trial_index, trial in enumerate(point["trials"]):
                selected_run = trial["final_selected_run"]
                if method == "papernot_baseline":
                    stage_1_E_K = point["E_K"]
                    stage_2_E_K = point["E_K"]
                    stage_1_sampled_K = trial["sampled_K"]
                    stage_2_sampled_K = trial["sampled_K"]
                else:
                    stage_1_E_K = point["stage_1_E_K"]
                    stage_2_E_K = point["stage_2_E_K"]
                    stage_1_sampled_K = trial[
                        "trial_stage_1"
                    ]["sampled_K"]
                    stage_2_sampled_K = trial[
                        "trial_stage_2"
                    ]["sampled_K"]

                for utility_metric in utility_metrics:
                    rows.append(
                        {
                            "method": method,
                            "point_index": point_index,
                            "trial": int(
                                trial.get("trial", trial_index)
                            ),
                            "expected_compute": point[
                                "expected_compute"
                            ],
                            "stage_1_E_K": stage_1_E_K,
                            "stage_2_E_K": stage_2_E_K,
                            "stage_1_sampled_K": (
                                stage_1_sampled_K
                            ),
                            "stage_2_sampled_K": (
                                stage_2_sampled_K
                            ),
                            "hp_configuration_id": selected_run[
                                "hp_configuration_id"
                            ],
                            "stage_1_run_index": selected_run[
                                "stage_1_run_index"
                            ],
                            "continuation_index": selected_run[
                                "continuation_index"
                            ],
                            "selection_metric": selected_run[
                                "selection"
                            ]["metric"],
                            "selection_mode": selected_run[
                                "selection"
                            ]["mode"],
                            "selection_stage": selected_run[
                                "selection"
                            ]["stage"],
                            "selection_round": selected_run[
                                "selection"
                            ]["round"],
                            "selection_score": selected_run[
                                "selection"
                            ]["score"],
                            "utility_metric": utility_metric,
                            "utility_score": selected_run[
                                "utility"
                            ][utility_metric],
                            "stage_1_metrics_path": selected_run[
                                "stage_1_metrics_path"
                            ],
                            "stage_2_metrics_path": selected_run[
                                "stage_2_metrics_path"
                            ],
                        }
                    )
    return rows


def save_trial_result_rows(rows, csv_path):
    if not rows:
        raise ValueError("No trial result rows were compiled.")
    with csv_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_expected_compute_utility(rows, output_directory):
    method_labels = {
        "papernot_baseline": "Papernot baseline",
        "two_stage_tuning": "Two-stage tuning",
    }
    utility_metrics = sorted(
        {row["utility_metric"] for row in rows}
    )
    plot_paths = []

    for utility_metric in utility_metrics:
        figure, axis = plt.subplots(figsize=(7, 5))
        for method in RESULT_FILENAMES:
            method_rows = [
                row
                for row in rows
                if row["method"] == method
                and row["utility_metric"] == utility_metric
            ]
            values_by_compute = {}
            for row in method_rows:
                values_by_compute.setdefault(
                    float(row["expected_compute"]),
                    [],
                ).append(float(row["utility_score"]))

            expected_compute_values = sorted(values_by_compute)
            means = []
            ci95_half_widths = []
            for expected_compute in expected_compute_values:
                values = np.asarray(
                    values_by_compute[expected_compute],
                    dtype=float,
                )
                standard_deviation = (
                    float(np.std(values, ddof=1))
                    if values.size > 1
                    else 0.0
                )
                means.append(float(np.mean(values)))
                ci95_half_widths.append(
                    1.96
                    * standard_deviation
                    / np.sqrt(values.size)
                )

            means = np.asarray(means, dtype=float)
            ci95_half_widths = np.asarray(
                ci95_half_widths,
                dtype=float,
            )
            mean_line, = axis.plot(
                expected_compute_values,
                means,
                marker="o",
                label=method_labels[method],
            )
            axis.fill_between(
                expected_compute_values,
                means - ci95_half_widths,
                means + ci95_half_widths,
                color=mean_line.get_color(),
                alpha=0.2,
                linewidth=0,
            )

        axis.set_xlabel(
            "Expected compute (communication rounds × local updates)"
        )
        axis.set_ylabel(
            utility_metric.replace("_", " ").title()
        )
        axis.set_title(
            f"{utility_metric.replace('_', ' ').title()} "
            "vs Expected Compute"
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        plot_path = (
            output_directory
            / f"expected_compute_vs_{utility_metric}.png"
        )
        figure.savefig(plot_path, dpi=300)
        plt.close(figure)
        plot_paths.append(plot_path)

    return plot_paths

def compile_experiment_results(config):
    evaluation = get_evaluation_settings(config)
    stage_compute_schedule = get_stage_compute_schedule(config)
    paths = get_compilation_paths(config)
    plans = {
        method: load_compilation_plan(config, method)
        for method in RESULT_FILENAMES
    }
    compiled_results = {
        "papernot_baseline": compile_papernot_results(
            config=config,
            plan=plans["papernot_baseline"],
            evaluation=evaluation,
            stage_compute_schedule=stage_compute_schedule,
        ),
        "two_stage_tuning": compile_two_stage_results(
            config=config,
            plan=plans["two_stage_tuning"],
            evaluation=evaluation,
            stage_compute_schedule=stage_compute_schedule,
        ),
    }

    compiled_root = paths["compiled_root"]
    compiled_root.mkdir(parents=True, exist_ok=True)
    result_paths = {}
    for method, result in compiled_results.items():
        plan_path = (
            paths["plan_root"]
            / PLAN_FILENAMES[method][2]
        )
        result["source_plan_path"] = str(plan_path)
        result_path = compiled_root / RESULT_FILENAMES[method]
        with result_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            encoder = json.JSONEncoder(
                indent=4,
                allow_nan=False,
            )
            pending_characters = 0

            for chunk in encoder.iterencode(result):
                file.write(chunk)
                pending_characters += len(chunk)

                if pending_characters >= 1_000_000:
                    file.flush()
                    pending_characters = 0
        result_paths[method] = result_path

    trial_rows = build_trial_result_rows(compiled_results)
    trial_csv_path = compiled_root / "trial_results.csv"
    save_trial_result_rows(
        rows=trial_rows,
        csv_path=trial_csv_path,
    )
    plot_paths = plot_expected_compute_utility(
        rows=trial_rows,
        output_directory=compiled_root,
    )

    return {
        "result_paths": result_paths,
        "trial_csv_path": trial_csv_path,
        "plot_paths": plot_paths,
    }

def calculate_E_K_given_compute_for_papernot(compute, local_updates_schedule):
    E_K = compute/sum(local_updates_schedule)
    return E_K


def calculate_compute_given_E_K_two_stage_tuning(config, E_K, local_updates_schedule):
    two_stage_settings = get_two_stage_settings(config)
    E_K_each_stage = [
        E_K,
        two_stage_settings.stage_2_expected_trials,
    ]
    compute = 0
    for i, E_K in enumerate(E_K_each_stage):
        compute += E_K * local_updates_schedule[i]
    return compute


def load_compiled_privacy_compute_points(config):
    """Load privacy coordinates from compiled output for validation only."""
    paths = get_compilation_paths(config)
    stage_compute_schedule = get_stage_compute_schedule(config)
    compiled_results = {}

    for method, filename in RESULT_FILENAMES.items():
        result_path = paths["compiled_root"] / filename
        if not result_path.is_file():
            raise FileNotFoundError(
                "Compiled results are required before privacy "
                f"accounting: {result_path}"
            )
        try:
            with result_path.open(
                mode="r",
                encoding="utf-8",
            ) as file:
                result = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Compiled results are not valid JSON: {result_path}"
            ) from error
        if result.get("method") != method:
            raise ValueError(
                f"Compiled result {result_path} has method "
                f"{result.get('method')!r}, expected {method!r}."
            )
        compiled_results[method] = result

    papernot_result = compiled_results["papernot_baseline"]
    two_stage_result = compiled_results["two_stage_tuning"]
    papernot_points = []
    for point_index, point in enumerate(
        papernot_result["points"]
    ):
        expected_num_trials = float(point["E_K"])
        expected_compute = float(point["expected_compute"])
        calculated_compute = (
            expected_num_trials * sum(stage_compute_schedule)
        )
        if not np.isclose(
            expected_compute,
            calculated_compute,
            rtol=1e-12,
            atol=1e-9,
        ):
            raise ValueError(
                "Papernot compiled expected compute is inconsistent "
                f"at point {point_index}: stored={expected_compute}, "
                f"calculated={calculated_compute}."
            )
        papernot_points.append(
            {
                "point_index": point_index,
                "expected_compute": expected_compute,
                "expected_num_trials": expected_num_trials,
            }
        )

    two_stage_points = []
    for point_index, point in enumerate(
        two_stage_result["points"]
    ):
        stage_1_expected_num_trials = float(
            point["stage_1_E_K"]
        )
        stage_2_expected_num_trials = float(
            point["stage_2_E_K"]
        )
        expected_compute = float(point["expected_compute"])
        calculated_compute = (
            stage_1_expected_num_trials
            * stage_compute_schedule[0]
            + stage_2_expected_num_trials
            * stage_compute_schedule[1]
        )
        if not np.isclose(
            expected_compute,
            calculated_compute,
            rtol=1e-12,
            atol=1e-9,
        ):
            raise ValueError(
                "Two-stage compiled expected compute is "
                f"inconsistent at point {point_index}: "
                f"stored={expected_compute}, "
                f"calculated={calculated_compute}."
            )
        two_stage_points.append(
            {
                "point_index": point_index,
                "expected_compute": expected_compute,
                "stage_1_expected_num_trials": (
                    stage_1_expected_num_trials
                ),
                "stage_2_expected_num_trials": (
                    stage_2_expected_num_trials
                ),
            }
        )

    papernot_points.sort(
        key=lambda point: point["expected_compute"]
    )
    two_stage_points.sort(
        key=lambda point: point["expected_compute"]
    )
    papernot_compute = np.asarray(
        [point["expected_compute"] for point in papernot_points]
    )
    two_stage_compute = np.asarray(
        [point["expected_compute"] for point in two_stage_points]
    )
    if (
        papernot_compute.shape != two_stage_compute.shape
        or not np.allclose(
            papernot_compute,
            two_stage_compute,
            rtol=1e-12,
            atol=1e-9,
        )
    ):
        raise ValueError(
            "Papernot and two-stage compiled results do not use "
            "matching expected-compute coordinates."
        )

    return {
        "papernot_points": papernot_points,
        "two_stage_points": two_stage_points,
        "papernot_eta": float(papernot_result["eta"]),
        "two_stage_eta": float(two_stage_result["eta"]),
        "two_stage_top_m": int(
            two_stage_result["stage_1_top_m"]
        ),
    }


def validate_privacy_order_search(dp_result, method, expected_compute):
    if dp_result.is_at_min_order or dp_result.is_at_max_order:
        boundary = (
            "minimum"
            if dp_result.is_at_min_order
            else "maximum"
        )
        raise RuntimeError(
            f"The optimal Rényi order for {method} at expected "
            f"compute {expected_compute} is the {boundary} stored "
            f"order ({dp_result.best_order}). Expand the configured "
            "Rényi-order range before reporting epsilon."
        )


def save_privacy_compute_rows(rows, csv_path):
    if not rows:
        raise ValueError("No privacy-compute result rows were produced.")
    with csv_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_privacy_compute_plot(config):
    exp_config = config.experiment
    privacy_config = exp_config.privacy
    max_renyi_order = int(privacy_config.max_renyi_order)
    if max_renyi_order < 3:
        raise ValueError(
            "privacy.max_renyi_order must be at least 3."
        )
    orders = np.arange(2, max_renyi_order + 1)
    delta = float(privacy_config.delta)
    if not 0.0 < delta < 1.0:
        raise ValueError(
            f"privacy.delta must satisfy 0 < delta < 1; got {delta}."
        )
    accounting_method = str(
        privacy_config.accounting_method
    ).strip().lower()
    if accounting_method not in {"bounds", "numerical"}:
        raise ValueError(
            "privacy.accounting_method must be 'bounds' or "
            f"'numerical'; got {accounting_method!r}."
        )

    low_resource_config = {
        "num_rounds": int(exp_config.simulation.stage_1_end),
        "num_local_updates": int(
            exp_config.local_updates_schedule[0]
        ),
        "num_clients": int(config.dataset.nb_users),
        "client_sampling_rate": float(config.server.client_ratio),
        "local_sampling_rate": float(config.server.sampling_rate),
        "sigma_gaussian": float(config.server.sigma),
        "sigma_is_actual": False,
    }
    high_resource_config = {
        "num_rounds": int(
            exp_config.simulation.stage_2_end
            - exp_config.simulation.stage_1_end
        ),
        "num_local_updates": int(
            exp_config.local_updates_schedule[1]
        ),
        "num_clients": int(config.dataset.nb_users),
        "client_sampling_rate": float(config.server.client_ratio),
        "local_sampling_rate": float(config.server.sampling_rate),
        "sigma_gaussian": float(config.server.sigma),
        "sigma_is_actual": False,
    }
    low_resource_curve = compute_dpfedavg_rdp(
        config=low_resource_config,
        orders=orders,
        accounting_method=accounting_method,
    )
    high_resource_curve = compute_dpfedavg_rdp(
        config=high_resource_config,
        orders=orders,
        accounting_method=accounting_method,
    )
    papernot_base_curve = compose_rdp_curves(
        low_resource_curve,
        high_resource_curve,
    )
    accounting_metadata = {
        "stage_1_num_rounds": low_resource_config[
            "num_rounds"
        ],
        "stage_2_num_rounds": high_resource_config[
            "num_rounds"
        ],
        "stage_1_num_local_updates": low_resource_config[
            "num_local_updates"
        ],
        "stage_2_num_local_updates": high_resource_config[
            "num_local_updates"
        ],
        "num_clients": low_resource_config["num_clients"],
        "client_sampling_rate": low_resource_config[
            "client_sampling_rate"
        ],
        "local_sampling_rate": low_resource_config[
            "local_sampling_rate"
        ],
        "sigma_gaussian": low_resource_config[
            "sigma_gaussian"
        ],
        "sigma_is_actual": low_resource_config[
            "sigma_is_actual"
        ],
        "effective_gaussian_noise_multiplier": (
            low_resource_config["sigma_gaussian"]
            * np.sqrt(
                low_resource_config["client_sampling_rate"]
                * low_resource_config["num_clients"]
            )
        ),
        "client_sampling_scheme": str(
            config.server.client_sampling_scheme
        ),
        "data_sampling_scheme": str(
            config.server.data_sampling_scheme
        ),
    }

    privacy_points = build_privacy_compute_points(
        exp_config,
        get_stage_compute_schedule(exp_config),
    )
    privacy_rows = []
    for point in privacy_points["papernot_points"]:
        selection_result = compute_top1_rdp(
            base_rdp_curve=papernot_base_curve,
            expected_num_trials=point["expected_num_trials"],
            eta=privacy_points["papernot_eta"],
        )
        dp_result = convert_rdp_to_approx_dp(
            selection_result.rdp_curve,
            delta=delta,
        )
        validate_privacy_order_search(
            dp_result=dp_result,
            method="papernot_baseline",
            expected_compute=point["expected_compute"],
        )
        privacy_rows.append(
            {
                "method": "papernot_baseline",
                "point_index": point["point_index"],
                "expected_compute": point["expected_compute"],
                "expected_num_trials": point[
                    "expected_num_trials"
                ],
                "stage_1_expected_num_trials": "",
                "stage_2_expected_num_trials": "",
                "top_m": 1,
                "eta": privacy_points["papernot_eta"],
                "epsilon": dp_result.epsilon,
                "delta": dp_result.delta,
                "best_renyi_order": dp_result.best_order,
                "is_at_min_order": dp_result.is_at_min_order,
                "is_at_max_order": dp_result.is_at_max_order,
                "min_renyi_order": int(orders[0]),
                "max_renyi_order": int(orders[-1]),
                "accounting_method": accounting_method,
                **accounting_metadata,
            }
        )

    for point in privacy_points["two_stage_points"]:
        selection_result = compute_two_stage_rdp(
            stage_1_base_rdp_curve=low_resource_curve,
            stage_2_base_rdp_curve=high_resource_curve,
            m=privacy_points["two_stage_top_m"],
            expected_num_trials_stage_1=point[
                "stage_1_expected_num_trials"
            ],
            expected_num_trials_stage_2=point[
                "stage_2_expected_num_trials"
            ],
            eta_stage_1=privacy_points["two_stage_eta"],
            eta_stage_2=privacy_points["two_stage_eta"],
        )
        dp_result = convert_rdp_to_approx_dp(
            selection_result.rdp_curve,
            delta=delta,
        )
        validate_privacy_order_search(
            dp_result=dp_result,
            method="two_stage_tuning",
            expected_compute=point["expected_compute"],
        )
        privacy_rows.append(
            {
                "method": "two_stage_tuning",
                "point_index": point["point_index"],
                "expected_compute": point["expected_compute"],
                "expected_num_trials": "",
                "stage_1_expected_num_trials": point[
                    "stage_1_expected_num_trials"
                ],
                "stage_2_expected_num_trials": point[
                    "stage_2_expected_num_trials"
                ],
                "top_m": privacy_points["two_stage_top_m"],
                "eta": privacy_points["two_stage_eta"],
                "epsilon": dp_result.epsilon,
                "delta": dp_result.delta,
                "best_renyi_order": dp_result.best_order,
                "is_at_min_order": dp_result.is_at_min_order,
                "is_at_max_order": dp_result.is_at_max_order,
                "min_renyi_order": int(orders[0]),
                "max_renyi_order": int(orders[-1]),
                "accounting_method": accounting_method,
                **accounting_metadata,
            }
        )

    paths = get_compilation_paths(exp_config)
    compiled_root = paths["compiled_root"]
    compiled_root.mkdir(parents=True, exist_ok=True)
    csv_path = compiled_root / "privacy_compute_results.csv"
    save_privacy_compute_rows(
        rows=privacy_rows,
        csv_path=csv_path,
    )

    method_labels = {
        "papernot_baseline": "Papernot baseline",
        "two_stage_tuning": "Two-stage tuning",
    }
    figure, axis = plt.subplots(figsize=(10, 5))
    for method in RESULT_FILENAMES:
        method_rows = sorted(
            (
                row
                for row in privacy_rows
                if row["method"] == method
            ),
            key=lambda row: row["expected_compute"],
        )
        axis.plot(
            [row["expected_compute"] for row in method_rows],
            [row["epsilon"] for row in method_rows],
            label=method_labels[method],
            marker="o",
        )

    axis.set_xlabel(
        "Expected compute (communication rounds × local updates)"
    )
    axis.set_ylabel(
        rf"$\varepsilon$ at $\delta={delta:.1e}$"
    )
    axis.set_title("Privacy-Compute Tradeoff")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    plot_path = compiled_root / "expected_compute_vs_privacy.png"
    figure.savefig(plot_path, dpi=300)
    plt.close(figure)

    return {
        "privacy_csv_path": csv_path,
        "plot_path": plot_path,
    }

def utility_compute_plot(config: DictConfig) -> None:
    exp_config = config.experiment
    two_stage_settings = get_two_stage_settings(exp_config)

    if exp_config.run_mode.generate_stage_1_plan:
        stage_compute_schedule = get_stage_compute_schedule(
            exp_config
        )
        privacy_points = build_privacy_compute_points(
            exp_config,
            stage_compute_schedule,
        )
        E_K_values_N_stage = [
            point["stage_1_expected_num_trials"]
            for point in privacy_points["two_stage_points"]
        ]
        E_K_values_papernot_baseline = [
            point["expected_num_trials"]
            for point in privacy_points["papernot_points"]
        ]
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
        plot_privacy_compute_plot(config)


    if exp_config.run_mode.run_simulation_stage_1:
        run_planned_simulations(
            config,
            stage=1,
        )

    if exp_config.run_mode.generate_stage_2_plan:
        plan_path = generate_stage_2_plan_from_results(exp_config)
        print(
            f"Generated Stage-2 plan: {plan_path}",
            flush=True,
        )

    if exp_config.run_mode.run_simulation_stage_2:
        run_planned_simulations(
            config,
            stage=2,
        )

    if exp_config.run_mode.compile_result:
        compile_experiment_results(exp_config)

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
