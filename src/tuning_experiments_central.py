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
    save_privacy_compute_rows,
    validate_privacy_order_search,
    generate_stage_2_plan_from_results,
)
from utils.hpo_config import get_two_stage_settings
from central.data import get_data_loaders
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
    compute_two_stage_rdp,
)
def _training_signature(config, stage, run_spec, learning_rate):
    """Return metadata that must match when a run is resumed or reused."""
    return {
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



def run_planned_simulations(config, stage):
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
    step_size = get_selected_learning_rate(config)
    simulations_root = (
        Path(HydraConfig.get().runtime.output_dir)
        / "simulations"
    )

    num_required_runs = len(required_run_specs)
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
                config=config,
                stage=stage,
                run_spec=run_spec,
                learning_rate=step_size,
            ),
        )
        OmegaConf.save(
            config,
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
                config=config,
                run_spec=run_spec,
                learning_rate=step_size,
                stage_1_source_path=stage_1_source_path,
            )

        train_data_loader, test_data_loader = get_data_loaders(
            config,
            seed=base_seed,
        )

        central_trainer = CentralTrainer(
            model=build_model(exp_config.dataset),
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
            num_iters=config.run_settings.rounds,
            save_path=save_path,
            loss_fn_name=config.experiment.dataset.loss_fn,
            learning_rate=step_size,
            weight_decay=config.run_settings.weight_decay,
            use_cuda=config.run_settings.use_cuda,
            dp=config.run_settings.dp,
            sample_rate=config.run_settings.sampling_rate,
            noise_multiplier=config.run_settings.noise_multiplier,
            max_grad_norm=config.run_settings.max_grad_norm,
            x_label=config.experiment.dataset.x_label,
            y_label=config.experiment.dataset.y_label,
            data_sampling_scheme=config.run_settings.data_sampling_scheme,
            stage=stage,
            stage_1_end=config.experiment.simulation.stage_1_end,
            base_seed=base_seed,
            stage_1_source_path=stage_1_source_path,
            optimizer_name=config.run_settings.optimizer_name,
            momentum=config.run_settings.momentum,
            checkpoint_interval=config.run_settings.checkpoint_interval,
            evaluation_interval=config.run_settings.evaluation_interval,
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

    if exp_config.run_mode.compile_result:
        raise NotImplementedError(
            "Central result compilation is not implemented yet. Do not "
            "enable experiment.run_mode.compile_result until the central "
            "trial compiler is available."
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
        plan_path = generate_stage_2_plan_from_results(
            exp_config,
            evaluation_interval=int(
                config.run_settings.evaluation_interval
            ),
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
        pass

EXPERIMENT_RUNNERS = {
    "utility_compute_plot": utility_compute_plot,
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
