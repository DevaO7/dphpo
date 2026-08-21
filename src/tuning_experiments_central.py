import hydra
import json
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from utils.seed_utils import set_global_seed
from hpo.execution import (
    get_selected_learning_rate,
    save_run_spec,
    validate_simulation_stage,
    validate_privacy_order_search,
)
from hpo.planning import (
    PLAN_FILENAMES,
    TWO_STAGE_METHODS,
    generate_plan,
    generate_privacy_matched_plan,
    get_privacy_matched_simulations_directory,
    get_required_simulation_run_specs,
    get_selection_signature,
    load_privacy_matched_simulation_plan,
    load_simulation_plan,
)
from hpo.results import (
    COMPUTE_MATCHED_METHOD_LABELS,
    COMPUTE_MATCHED_RESULT_FILENAMES,
    build_privacy_compute_points,
    compile_experiment_results,
    compile_privacy_matched_results,
    generate_privacy_matched_stage_2_plan_from_results,
    generate_stage_2_plan_from_results,
    get_compute_matched_methods,
    get_compilation_paths,
    save_privacy_compute_rows,
)
from utils.hpo_config import get_two_stage_settings
from central.data import get_data_loaders
from central.evaluation import HeldoutCheckpointEvaluator
from central.model import build_model
from central.trainer import CentralTrainer
import matplotlib.pyplot as plt
from privacy_accounting.dpsgd import compute_dpsgd_rdp
from privacy_accounting.rdp_utils import (
    compose_rdp_curves,
    convert_rdp_to_approx_dp,
)
from privacy_accounting.selection_accounting import (
    compute_top1_rdp,
    compute_top1_rdp_poisson,
    compute_two_stage_rdp,
    compute_two_stage_rdp_poisson,
)
def _training_signature(config, stage, run_spec, learning_rate):
    """Return metadata that must match when a run is resumed or reused."""
    signature = {
        "schema_version": 1,
        "stage": int(stage),
        "base_seed": int(run_spec[f"stage_{stage}_base_seed"]),
        "stage_1_end": int(config.experiment.simulation.stage_1_end),
        "num_iters": int(config.run_settings.rounds),
        "learning_rate": float(learning_rate),
        "dataset": OmegaConf.to_container(
            config.experiment.dataset,
            resolve=True,
        ),
        "run_settings": OmegaConf.to_container(
            config.run_settings,
            resolve=True,
        ),
    }
    utility_at = str(
        config.experiment.evaluation.utility.get(
            "at",
            "selection_round",
        )
    ).strip().lower()
    if utility_at == "selected_checkpoint":
        signature["schema_version"] = 2
        signature["selection_signature"] = get_selection_signature(
            config
        )
    return signature


def _save_or_validate_training_signature(
    save_path,
    stage,
    signature,
):
    """Persist immutable training metadata before creating a checkpoint."""
    signature_path = Path(save_path) / (
        f"stage_{stage}_training_signature.JSON"
    )
    if signature_path.is_file():
        try:
            with signature_path.open(encoding="utf-8") as file:
                existing_signature = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Training signature is invalid: {signature_path}"
            ) from error
        if existing_signature != signature:
            raise ValueError(
                "The existing run was created with different training "
                f"settings: {signature_path}. Use a new run_id or "
                "restore the original settings before resuming."
            )
        return signature_path

    checkpoint_path = Path(save_path) / f"stage_{stage}.pt"
    if checkpoint_path.is_file():
        raise RuntimeError(
            "A checkpoint exists without the immutable training "
            f"signature required to validate it: {checkpoint_path}. "
            "Use a new run_id or explicitly migrate the trusted legacy "
            "checkpoint before resuming."
        )

    temporary_path = signature_path.with_suffix(".JSON.tmp")
    try:
        with temporary_path.open(mode="w", encoding="utf-8") as file:
            json.dump(signature, file, indent=4, allow_nan=False)
        temporary_path.replace(signature_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return signature_path


def _validate_stage_1_source_signature(
    config,
    run_spec,
    learning_rate,
    stage_1_source_path,
):
    expected_signature = _training_signature(
        config=config,
        stage=1,
        run_spec=run_spec,
        learning_rate=learning_rate,
    )
    stage_1_end = int(config.experiment.simulation.stage_1_end)
    expected_signature["num_iters"] = stage_1_end
    expected_signature["run_settings"]["rounds"] = stage_1_end
    signature_path = (
        Path(stage_1_source_path) / "stage_1_training_signature.JSON"
    )
    if not signature_path.is_file():
        raise FileNotFoundError(
            "Cannot start Stage 2 because the Stage-1 training "
            f"signature does not exist: {signature_path}"
        )
    try:
        with signature_path.open(encoding="utf-8") as file:
            observed_signature = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Stage-1 training signature is invalid: {signature_path}"
        ) from error
    if observed_signature != expected_signature:
        raise ValueError(
            "The Stage-1 source was trained with settings that do not "
            f"match this Stage-2 run: {signature_path}."
        )

def load_compiled_privacy_compute_points(config):
    """Load privacy coordinates from compiled output for validation only."""
    paths = get_compilation_paths(config)
    stage_compute_schedule = get_stage_compute_schedule(config)
    methods = get_compute_matched_methods(config)
    compiled_results = {}

    for method in methods:
        filename = COMPUTE_MATCHED_RESULT_FILENAMES[method]
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

    points_by_method = {}
    for method, result in compiled_results.items():
        method_points = []
        for point_index, point in enumerate(result["points"]):
            expected_compute = float(point["expected_compute"])
            if method in TWO_STAGE_METHODS:
                stage_1_expected_num_trials = float(
                    point["stage_1_E_K"]
                )
                stage_2_expected_num_trials = float(
                    point["stage_2_E_K"]
                )
                calculated_compute = (
                    stage_1_expected_num_trials
                    * stage_compute_schedule[0]
                    + stage_2_expected_num_trials
                    * stage_compute_schedule[1]
                )
                method_point = {
                    "point_index": point_index,
                    "expected_compute": expected_compute,
                    "stage_1_expected_num_trials": (
                        stage_1_expected_num_trials
                    ),
                    "stage_2_expected_num_trials": (
                        stage_2_expected_num_trials
                    ),
                }
            else:
                expected_num_trials = float(point["E_K"])
                calculated_compute = expected_num_trials * sum(
                    stage_compute_schedule
                )
                method_point = {
                    "point_index": point_index,
                    "expected_compute": expected_compute,
                    "expected_num_trials": expected_num_trials,
                }
            if not np.isclose(
                expected_compute,
                calculated_compute,
                rtol=1e-12,
                atol=1e-9,
            ):
                raise ValueError(
                    f"{method} compiled expected compute is inconsistent "
                    f"at point {point_index}: stored={expected_compute}, "
                    f"calculated={calculated_compute}."
                )
            method_points.append(method_point)
        method_points.sort(key=lambda point: point["expected_compute"])
        points_by_method[method] = method_points

    two_stage_points = points_by_method["two_stage_tuning"]
    two_stage_compute = np.asarray(
        [point["expected_compute"] for point in two_stage_points]
    )
    for method, method_points in points_by_method.items():
        method_compute = np.asarray(
            [point["expected_compute"] for point in method_points]
        )
        if (
            method_compute.shape != two_stage_compute.shape
            or not np.allclose(
                method_compute,
                two_stage_compute,
                rtol=1e-12,
                atol=1e-9,
            )
        ):
            raise ValueError(
                f"{method} and two_stage_tuning compiled results do not "
                "use matching expected-compute coordinates."
            )

    return {
        "points_by_method": points_by_method,
        "papernot_points": points_by_method.get(
            "papernot_baseline",
            [],
        ),
        "papernot_poisson_points": points_by_method.get(
            "papernot_poisson_baseline",
            [],
        ),
        "two_stage_points": two_stage_points,
        "papernot_eta": float(
            compiled_results.get(
                "papernot_baseline",
                compiled_results["two_stage_tuning"],
            )["eta"]
        ),
        "two_stage_eta": float(
            compiled_results["two_stage_tuning"]["eta"]
        ),
        "two_stage_top_m": int(
            compiled_results["two_stage_tuning"]["stage_1_top_m"]
        ),
    }


def plot_privacy_compute_plot(config):
    exp_config = config.experiment
    methods = get_compute_matched_methods(exp_config)
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
    accounting_method = "numerical"

    low_resource_config = {
        "num_rounds": int(exp_config.simulation.stage_1_end),
        "data_sampling_rate": float(config.run_settings.sampling_rate),
        "sigma_gaussian": float(config.run_settings.noise_multiplier),
    }
    high_resource_config = {
        "num_rounds": int(exp_config.simulation.stage_2_end)-int(exp_config.simulation.stage_1_end),
        "data_sampling_rate": float(config.run_settings.sampling_rate),
        "sigma_gaussian": float(config.run_settings.noise_multiplier),
    }
    low_resource_curve = compute_dpsgd_rdp(
        config=low_resource_config,
        orders=orders,
    )
    high_resource_curve = compute_dpsgd_rdp(
        config=high_resource_config,
        orders=orders,
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
        "data_sampling_rate": low_resource_config[
            "data_sampling_rate"
        ],
        "sigma_gaussian": low_resource_config[
            "sigma_gaussian"
        ]
    }
    privacy_points = build_privacy_compute_points(
        exp_config,
        get_stage_compute_schedule(exp_config),
    )
    privacy_rows = []
    for method in (
        method
        for method in methods
        if method in {
            "papernot_baseline",
            "papernot_poisson_baseline",
        }
    ):
        for point in privacy_points["papernot_points"]:
            expected_num_trials = point["expected_num_trials"]
            if method == "papernot_poisson_baseline":
                selection_result = compute_top1_rdp_poisson(
                    base_rdp_curve=papernot_base_curve,
                    expected_num_trials=expected_num_trials,
                )
            else:
                selection_result = compute_top1_rdp(
                    base_rdp_curve=papernot_base_curve,
                    expected_num_trials=expected_num_trials,
                    eta=privacy_points["papernot_eta"],
                )
            dp_result = convert_rdp_to_approx_dp(
                selection_result.rdp_curve,
                delta=delta,
            )
            validate_privacy_order_search(
                dp_result=dp_result,
                method=method,
                expected_compute=point["expected_compute"],
            )
            poisson_diagnostics = {
                "theorem_source_order": "",
                "hat_epsilon": "",
                "hat_delta": "",
                "best_auxiliary_renyi_order": "",
                "raw_theorem_rdp_epsilon": "",
                "enveloped_rdp_epsilon": "",
            }
            if method == "papernot_poisson_baseline":
                target_index = int(dp_result.best_index)
                source_order = float(
                    selection_result.envelope_source_orders[
                        target_index
                    ]
                )
                source_index = int(
                    np.flatnonzero(
                        np.isclose(
                            selection_result.raw_rdp_curve.orders,
                            source_order,
                            rtol=0.0,
                            atol=1e-12,
                        )
                    )[0]
                )
                poisson_diagnostics = {
                    "theorem_source_order": source_order,
                    "hat_epsilon": float(
                        selection_result.hat_epsilons[source_index]
                    ),
                    "hat_delta": float(
                        selection_result.hat_deltas[source_index]
                    ),
                    "best_auxiliary_renyi_order": float(
                        selection_result.best_auxiliary_orders[
                            source_index
                        ]
                    ),
                    "raw_theorem_rdp_epsilon": float(
                        selection_result.raw_rdp_curve.epsilons[
                            source_index
                        ]
                    ),
                    "enveloped_rdp_epsilon": float(
                        selection_result.rdp_curve.epsilons[
                            target_index
                        ]
                    ),
                }
            privacy_rows.append(
                {
                    "method": method,
                    "point_index": point["point_index"],
                    "expected_compute": point["expected_compute"],
                    "expected_num_trials": expected_num_trials,
                    "stage_1_expected_num_trials": "",
                    "stage_2_expected_num_trials": "",
                    "top_m": 1,
                    "sampling_distribution": (
                        "poisson"
                        if method == "papernot_poisson_baseline"
                        else "tnb"
                    ),
                    "eta": (
                        ""
                        if method == "papernot_poisson_baseline"
                        else privacy_points["papernot_eta"]
                    ),
                    "probability_K_zero": (
                        float(np.exp(-expected_num_trials))
                        if method == "papernot_poisson_baseline"
                        else 0.0
                    ),
                    "epsilon": dp_result.epsilon,
                    "delta": dp_result.delta,
                    "best_renyi_order": dp_result.best_order,
                    "is_at_min_order": dp_result.is_at_min_order,
                    "is_at_max_order": dp_result.is_at_max_order,
                    "min_renyi_order": int(orders[0]),
                    "max_renyi_order": int(orders[-1]),
                    "accounting_method": accounting_method,
                    **poisson_diagnostics,
                    **accounting_metadata,
                }
            )

    for method in (
        method
        for method in methods
        if method in TWO_STAGE_METHODS
    ):
        for point in privacy_points["two_stage_points"]:
            common_arguments = {
                "stage_1_base_rdp_curve": low_resource_curve,
                "stage_2_base_rdp_curve": high_resource_curve,
                "m": privacy_points["two_stage_top_m"],
                "expected_num_trials_stage_1": point[
                    "stage_1_expected_num_trials"
                ],
                "expected_num_trials_stage_2": point[
                    "stage_2_expected_num_trials"
                ],
            }
            if method == "two_stage_poisson_tuning":
                selection_result = compute_two_stage_rdp_poisson(
                    **common_arguments
                )
                sampling_distribution = "poisson"
                eta = ""
                probability_k_zero = float(
                    np.exp(-point["stage_2_expected_num_trials"])
                )
            else:
                selection_result = compute_two_stage_rdp(
                    **common_arguments,
                    eta_stage_1=privacy_points["two_stage_eta"],
                    eta_stage_2=privacy_points["two_stage_eta"],
                )
                sampling_distribution = "tnb"
                eta = privacy_points["two_stage_eta"]
                probability_k_zero = 0.0
            dp_result = convert_rdp_to_approx_dp(
                selection_result.rdp_curve,
                delta=delta,
            )
            validate_privacy_order_search(
                dp_result=dp_result,
                method=method,
                expected_compute=point["expected_compute"],
            )
            privacy_rows.append(
                {
                    "method": method,
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
                    "sampling_distribution": sampling_distribution,
                    "eta": eta,
                    "probability_K_zero": probability_k_zero,
                    "epsilon": dp_result.epsilon,
                    "delta": dp_result.delta,
                    "best_renyi_order": dp_result.best_order,
                    "is_at_min_order": dp_result.is_at_min_order,
                    "is_at_max_order": dp_result.is_at_max_order,
                    "min_renyi_order": int(orders[0]),
                    "max_renyi_order": int(orders[-1]),
                    "accounting_method": accounting_method,
                    "theorem_source_order": "",
                    "hat_epsilon": "",
                    "hat_delta": "",
                    "best_auxiliary_renyi_order": "",
                    "raw_theorem_rdp_epsilon": "",
                    "enveloped_rdp_epsilon": "",
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

    method_styles = {
        "papernot_baseline": {"linestyle": "-", "marker": "o"},
        "papernot_poisson_baseline": {
            "linestyle": ":",
            "marker": "^",
        },
        "two_stage_tuning": {"linestyle": "--", "marker": "s"},
        "two_stage_poisson_tuning": {
            "linestyle": "-.",
            "marker": "D",
        },
    }
    figure, axis = plt.subplots(figsize=(10, 5))
    for method in methods:
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
            label=COMPUTE_MATCHED_METHOD_LABELS[method],
            **method_styles[method],
        )

    axis.set_xlabel("Expected compute (optimizer updates)")
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



def _config_with_plan_noise_multiplier(config, simulation_plan):
    """Copy the config and install the plan-calibrated noise value."""
    points = simulation_plan.get("points")
    if not isinstance(points, list) or len(points) != 1:
        raise ValueError(
            "A privacy-matched simulation plan must contain exactly "
            "one point."
        )
    noise_multiplier = float(points[0]["noise_multiplier"])
    if not np.isfinite(noise_multiplier) or noise_multiplier <= 0.0:
        raise ValueError(
            "The plan noise_multiplier must be finite and positive."
        )

    effective_config = OmegaConf.create(
        OmegaConf.to_container(config, resolve=True)
    )
    effective_config.run_settings.noise_multiplier = noise_multiplier
    return effective_config


def run_planned_simulations(config, stage, *, privacy_matched=False):
    validate_simulation_stage(config, stage)
    if (
        bool(config.run_settings.dp)
        and str(config.run_settings.data_sampling_scheme)
        != "poisson_sampling"
    ):
        raise ValueError(
            "Central DP-SGD experiments currently require "
            "run_settings.data_sampling_scheme='poisson_sampling' so "
            "training matches the privacy accountant."
        )
    if privacy_matched:
        simulation_plan = load_privacy_matched_simulation_plan(
            config,
            stage=stage,
        )
        execution_config = _config_with_plan_noise_multiplier(
            config,
            simulation_plan,
        )
        exp_config = execution_config.experiment
        simulations_root = get_privacy_matched_simulations_directory(
            config=execution_config,
            method=str(exp_config.simulation.method),
            target_epsilon=float(
                exp_config.simulation.target_epsilon
            ),
            E_k=float(exp_config.simulation.mu),
        )
    else:
        execution_config = config
        exp_config = execution_config.experiment
        utility_at = str(
            exp_config.evaluation.utility.get(
                "at",
                "selection_round",
            )
        ).strip().lower()
        simulation_plan = load_simulation_plan(
            exp_config,
            stage=stage,
            selection_signature=(
                get_selection_signature(config)
                if utility_at == "selected_checkpoint"
                else None
            ),
        )
        simulations_root = (
            Path(HydraConfig.get().runtime.output_dir)
            / "simulations"
        )

    required_run_specs = get_required_simulation_run_specs(
        simulation_plan,
        stage=stage,
        hp_configuration_id=(
            exp_config.simulation.run_hp_configuration
        ),
    )
    step_size = get_selected_learning_rate(execution_config)

    num_required_runs = len(required_run_specs)
    if privacy_matched:
        print(
            "Using plan-calibrated noise_multiplier="
            f"{execution_config.run_settings.noise_multiplier}",
            flush=True,
        )
    print(
        f"Stage {stage} {exp_config.simulation.run_hp_configuration}: "
        f"{num_required_runs} plan-defined runs",
        flush=True,
    )
    for run_number, run_spec in enumerate(required_run_specs, start=1):
        base_seed = int(
            run_spec[f"stage_{stage}_base_seed"]
        )
        set_global_seed(base_seed)
        save_path = (
            simulations_root
            / run_spec[f"stage_{stage}_run_directory"]
        )
        save_path.mkdir(parents=True, exist_ok=True)
        print(
            f"Starting Stage {stage} run {run_number}/"
            f"{num_required_runs}: {save_path}",
            flush=True,
        )
        save_run_spec(
            save_path=save_path,
            stage=stage,
            run_spec=run_spec,
        )
        _save_or_validate_training_signature(
            save_path=save_path,
            stage=stage,
            signature=_training_signature(
                config=execution_config,
                stage=stage,
                run_spec=run_spec,
                learning_rate=step_size,
            ),
        )
        OmegaConf.save(
            execution_config,
            save_path / f"stage_{stage}_config.yaml",
            resolve=True,
        )

        stage_1_source_path = None
        if stage == 2:
            stage_1_source_path = (
                simulations_root
                / run_spec["stage_1_run_directory"]
            )
            _validate_stage_1_source_signature(
                config=execution_config,
                run_spec=run_spec,
                learning_rate=step_size,
                stage_1_source_path=stage_1_source_path,
            )

        utility_at = str(
            exp_config.evaluation.utility.get(
                "at",
                "selection_round",
            )
        ).strip().lower()
        peak_aware = utility_at == "selected_checkpoint"
        if peak_aware:
            (
                train_data_loader,
                validation_data_loader,
                _,
            ) = get_data_loaders(
                execution_config,
                seed=base_seed,
                split_public_evaluation=True,
            )
            test_data_loader = None
        else:
            train_data_loader, test_data_loader = get_data_loaders(
                execution_config,
                seed=base_seed,
            )
            validation_data_loader = None

        central_trainer = CentralTrainer(
            model=build_model(exp_config.dataset),
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
            num_iters=execution_config.run_settings.rounds,
            save_path=save_path,
            loss_fn_name=execution_config.experiment.dataset.loss_fn,
            learning_rate=step_size,
            weight_decay=execution_config.run_settings.weight_decay,
            use_cuda=execution_config.run_settings.use_cuda,
            dp=execution_config.run_settings.dp,
            sample_rate=execution_config.run_settings.sampling_rate,
            noise_multiplier=(
                execution_config.run_settings.noise_multiplier
            ),
            max_grad_norm=execution_config.run_settings.max_grad_norm,
            x_label=execution_config.experiment.dataset.x_label,
            y_label=execution_config.experiment.dataset.y_label,
            data_sampling_scheme=(
                execution_config.run_settings.data_sampling_scheme
            ),
            stage=stage,
            stage_1_end=(
                execution_config.experiment.simulation.stage_1_end
            ),
            base_seed=base_seed,
            stage_1_source_path=stage_1_source_path,
            optimizer_name=execution_config.run_settings.optimizer_name,
            momentum=execution_config.run_settings.momentum,
            checkpoint_interval=(
                execution_config.run_settings.checkpoint_interval
            ),
            evaluation_interval=(
                execution_config.run_settings.evaluation_interval
            ),
            validation_data_loader=validation_data_loader,
            selection_metric=(
                str(exp_config.evaluation.selection.metric)
                if peak_aware
                else None
            ),
            selection_mode=(
                str(exp_config.evaluation.selection.mode)
                if peak_aware
                else None
            ),
            peak_tie_break=str(
                exp_config.evaluation.selection.get(
                    "peak_tie_break",
                    "earliest",
                )
            ),
        )
        central_trainer.train()
        print(
            f"Completed Stage {stage} run {run_number}/"
            f"{num_required_runs}: {save_path}",
            flush=True,
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
    methods = get_compute_matched_methods(exp_config)
    utility_at = str(
        exp_config.evaluation.utility.get("at", "selection_round")
    ).strip().lower()
    selection_signature = (
        get_selection_signature(config)
        if utility_at == "selected_checkpoint"
        else None
    )
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
        for method in methods:
            is_two_stage = method in TWO_STAGE_METHODS
            generate_plan(
                exp_config,
                method,
                (
                    two_stage_settings.num_survivors
                    if is_two_stage
                    else 1
                ),
                (
                    E_K_values_N_stage
                    if is_two_stage
                    else E_K_values_papernot_baseline
                ),
                exp_config.num_trials,
                exp_config.run_id,
                exp_config.hp_configuration_ids,
                PLAN_FILENAMES[method][1],
                plan_metadata={
                    "plan_type": "compute_matched",
                    **(
                        {"selection_signature": selection_signature}
                        if selection_signature is not None
                        else {}
                    ),
                },
            )
        plot_privacy_compute_plot(config)

    if exp_config.run_mode.run_simulation_stage_1:
        run_planned_simulations(
            config,
            stage=1,
        )

    if exp_config.run_mode.generate_stage_2_plan:
        plan_path = generate_stage_2_plan_from_results(
            exp_config,
            evaluation_interval=int(
                config.run_settings.evaluation_interval
            ),
            selection_signature=selection_signature,
        )
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
        utility_evaluator = (
            HeldoutCheckpointEvaluator(config)
            if utility_at == "selected_checkpoint"
            else None
        )
        outputs = compile_experiment_results(
            exp_config,
            stage_compute_schedule=get_stage_compute_schedule(
                exp_config
            ),
            evaluation_interval=int(
                config.run_settings.evaluation_interval
            ),
            compute_axis_label=(
                "Expected compute (optimizer updates)"
            ),
            utility_evaluator=utility_evaluator,
            selection_signature=selection_signature,
        )
        print(
            f"Compiled central results: {outputs['trial_csv_path']}",
            flush=True,
        )


def utility_privacy_plot(config: DictConfig) -> None:
    exp_config = config.experiment
    method = str(exp_config.simulation.method)
    E_k = float(exp_config.simulation.mu)
    target_epsilon = float(
        exp_config.simulation.target_epsilon
    )

    if exp_config.run_mode.generate_stage_1_plan:
        plan_path = generate_privacy_matched_plan(
            config=config,
            method=method,
            target_epsilon=target_epsilon,
            E_k=E_k,
        )
        print(
            f"Generated privacy-matched Stage-1 plan: {plan_path}",
            flush=True,
        )

    if exp_config.run_mode.run_simulation_stage_1:
        run_planned_simulations(
            config,
            stage=1,
            privacy_matched=True,
        )

    if exp_config.run_mode.generate_stage_2_plan:
        plan_path = (
            generate_privacy_matched_stage_2_plan_from_results(
                config,
                evaluation_interval=int(
                    config.run_settings.evaluation_interval
                ),
            )
        )
        print(
            "Generated privacy-matched Stage-2 plan: "
            f"{plan_path}",
            flush=True,
        )

    if exp_config.run_mode.run_simulation_stage_2:
        run_planned_simulations(
            config,
            stage=2,
            privacy_matched=True,
        )

    if exp_config.run_mode.compile_result:
        utility_evaluator = None
        compilation_targets = [
            str(target)
            for target in exp_config.compilation.targets
        ]
        if "utility_privacy" in compilation_targets:
            utility_evaluator = HeldoutCheckpointEvaluator(config)
        outputs = compile_privacy_matched_results(
            config,
            utility_evaluator=utility_evaluator,
        )
        for target, target_outputs in outputs.items():
            print(
                "Compiled privacy-matched result "
                f"{target}: {target_outputs['output_directory']}",
                flush=True,
            )


EXPERIMENT_RUNNERS = {
    "utility_compute_plot": utility_compute_plot,
    "utility_privacy_plot": utility_privacy_plot,
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
