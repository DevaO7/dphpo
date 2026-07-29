import numpy as np
import matplotlib.pyplot as plt
import hydra
import csv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from privacy_accounting.tnb import _solve_gamma_for_conditional_mean, TNBDistribution
from pathlib import Path
import json
from collections import Counter
from utils.data_utils import get_data_loaders, set_seed
from flearn.trainmodel import models
from flearn.servers.server_avg import FedAvg


def get_selected_learning_rate(config: DictConfig) -> float:
    exp_config = config.experiment
    hp_configuration_id = str(
        exp_config.simulation.run_hp_configuration
    )

    if hp_configuration_id not in config.hp_candidate_set:
        available_ids = ", ".join(
            config.hp_candidate_set.keys()
        )
        raise ValueError(
            "Unknown hyperparameter configuration "
            f"{hp_configuration_id!r}. Available configurations: "
            f"{available_ids}"
        )

    hp_configuration = config.hp_candidate_set[
        hp_configuration_id
    ]

    if "step_size" not in hp_configuration:
        raise ValueError(
            "Hyperparameter configuration "
            f"{hp_configuration_id!r} does not define "
            "'step_size'."
        )

    learning_rate = float(
        hp_configuration.step_size
    )

    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(
            f"Hyperparameter configuration {hp_configuration_id!r} "
            "must define a finite, positive 'step_size'; "
            f"got {learning_rate!r}."
        )

    return learning_rate


def generate_plan(
    config,
    m,
    E_K_values,
    num_trials,
    run_id,
    hp_configuration_ids,
    plan_filename,
):
    # Total number of appearances across the complete plan.
    hp_configuration_counts = {
        hp_id: 0
        for hp_id in hp_configuration_ids
    }

    # Maximum number of appearances within any single sampled HP list.
    required_hp_configuration_runs = {
        hp_id: 0
        for hp_id in hp_configuration_ids
    }

    points = []
    total_num_simulations = 0

    for point_index, E_K in enumerate(E_K_values):
        gamma = _solve_gamma_for_conditional_mean(
            eta=config.eta,
            m=m,
            target_mean=E_K,
        )

        tnb = TNBDistribution(
            config.eta,
            gamma,
        )

        trials = []

        for trial in range(num_trials):
            rng = np.random.default_rng(
                seed=trial + config.seed + point_index
            )

            num_runs = int(
                tnb.sample_conditional(m, rng)
            )

            sampled_hp_configuration_ids = (
                rng.choice(
                    hp_configuration_ids,
                    size=num_runs,
                    replace=True,
                )
                .tolist()
            )

            trials.append(
                {
                    "trial": trial,
                    "sampled_K": num_runs,
                    "sampled_hp_configuration_ids": (
                        sampled_hp_configuration_ids
                    ),
                }
            )

            total_num_simulations += num_runs

            # Count multiplicities within this particular
            # E[K]-trial combination.
            current_hp_counts = Counter(
                sampled_hp_configuration_ids
            )

            for hp_id in hp_configuration_ids:
                current_count = current_hp_counts.get(
                    hp_id,
                    0,
                )

                # Cumulative planned appearances.
                hp_configuration_counts[
                    hp_id
                ] += current_count

                # Global maximum multiplicity.
                required_hp_configuration_runs[
                    hp_id
                ] = max(
                    required_hp_configuration_runs[hp_id],
                    current_count,
                )

        points.append(
            {
                "E_K": float(E_K),
                "gamma": float(gamma),
                "trials": trials,
            }
        )

    total_num_required_runs = sum(
        required_hp_configuration_runs.values()
    )

    plan = {
        "method": "papernot_top1",
        "eta": config.eta,
        "plan_seed": run_id,
        "points": points,
        "execution_summary": {
            # Number of HP occurrences in the complete plan.
            "total_num_simulations": total_num_simulations,
            "hp_configuration_counts": (
                hp_configuration_counts
            ),

            # Number of actual DP-FedAvg trajectories that will
            # be generated under the global-maximum reuse scheme.
            "total_num_required_runs": total_num_required_runs,
            "required_hp_configuration_runs": (
                required_hp_configuration_runs
            ),
        },
    }

    results_root = Path(
        config.output.results_root
    )

    experiment_name = str(
        config.name
    )

    plan_directory = (
        results_root
        / experiment_name
        / str(run_id)
        / "plan"
    )

    plan_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plan_path = (
        plan_directory
        / plan_filename
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


def load_stage_1_plan(
    config,
    plan_filename="two_stage_tuning_stage_1.JSON",
):
    plan_path = (
        Path(config.output.results_root)
        / str(config.name)
        / str(config.run_id)
        / "plan"
        / plan_filename
    )

    if not plan_path.is_file():
        raise FileNotFoundError(
            "Stage-1 plan does not exist: "
            f"{plan_path}"
        )

    try:
        with plan_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            plan = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Stage-1 plan is not valid JSON: {plan_path}"
        ) from error

    if not isinstance(plan, dict):
        raise ValueError(
            "Stage-1 plan must contain a JSON object at its root: "
            f"{plan_path}"
        )

    if not isinstance(plan.get("points"), list):
        raise ValueError(
            "Stage-1 plan must contain a 'points' list: "
            f"{plan_path}"
        )

    if not isinstance(plan.get("execution_summary"), dict):
        raise ValueError(
            "Stage-1 plan must contain an "
            "'execution_summary' object: "
            f"{plan_path}"
        )

    return plan


def map_stage_1_plan_runs(
    stage_1_plan,
    hp_configuration_ids,
):
    known_hp_configuration_ids = {
        str(hp_id)
        for hp_id in hp_configuration_ids
    }

    for point_index, point in enumerate(
        stage_1_plan["points"]
    ):
        if not isinstance(point, dict):
            raise ValueError(
                "Each Stage-1 plan point must be an object; "
                f"point {point_index} is invalid."
            )

        trials = point.get("trials")
        if not isinstance(trials, list):
            raise ValueError(
                "Each Stage-1 plan point must contain a 'trials' "
                f"list; point {point_index} is invalid."
            )

        for trial_index, trial in enumerate(trials):
            if not isinstance(trial, dict):
                raise ValueError(
                    "Each Stage-1 trial must be an object; "
                    f"point {point_index}, trial {trial_index} "
                    "is invalid."
                )

            sampled_hp_ids = trial.get(
                "sampled_hp_configuration_ids"
            )
            if not isinstance(sampled_hp_ids, list):
                raise ValueError(
                    "Each Stage-1 trial must contain a "
                    "'sampled_hp_configuration_ids' list; "
                    f"point {point_index}, trial {trial_index} "
                    "is invalid."
                )

            sampled_k = trial.get("sampled_K")
            if sampled_k != len(sampled_hp_ids):
                raise ValueError(
                    "Stage-1 sampled_K does not match the number "
                    "of sampled hyperparameter configurations at "
                    f"point {point_index}, trial {trial_index}: "
                    f"sampled_K={sampled_k!r}, "
                    f"list length={len(sampled_hp_ids)}."
                )

            hp_occurrence_counts = Counter()
            sampled_stage_1_runs = []

            for sample_index, hp_id in enumerate(sampled_hp_ids):
                hp_id = str(hp_id)
                if hp_id not in known_hp_configuration_ids:
                    raise ValueError(
                        "Unknown hyperparameter configuration "
                        f"{hp_id!r} at point {point_index}, "
                        f"trial {trial_index}, sample "
                        f"{sample_index}."
                    )

                stage_1_seed = hp_occurrence_counts[hp_id]
                hp_occurrence_counts[hp_id] += 1

                sampled_stage_1_runs.append(
                    {
                        "sample_index": sample_index,
                        "hp_configuration_id": hp_id,
                        "stage_1_seed": stage_1_seed,
                        "stage_1_run_directory": (
                            f"{hp_id}/seed_{stage_1_seed}"
                        ),
                    }
                )

            trial["sampled_stage_1_runs"] = (
                sampled_stage_1_runs
            )

    return stage_1_plan


def load_stage_1_metric(
    csv_path,
    metric,
    evaluation_mode,
    expected_final_round,
):
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Stage-1 metrics CSV does not exist: {csv_path}"
        )

    try:
        with csv_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError(
                    f"Stage-1 metrics CSV has no header: {csv_path}"
                )

            normalized_columns = {
                column.strip().lower().replace(" ", "_"): column
                for column in fieldnames
            }
            normalized_metric = (
                str(metric).strip().lower().replace(" ", "_")
            )

            try:
                metric_column = normalized_columns[
                    normalized_metric
                ]
                round_column = normalized_columns["round"]
            except KeyError as error:
                available_metrics = ", ".join(fieldnames)
                raise ValueError(
                    f"Metric {metric!r} is not available in "
                    f"{csv_path}. Available columns: "
                    f"{available_metrics}."
                ) from error

            metric_rows = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    round_number = int(row[round_column])
                    score = float(row[metric_column])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Stage-1 metrics CSV contains an invalid "
                        f"round or score at row {row_number}: "
                        f"{csv_path}."
                    ) from error

                if not np.isfinite(score):
                    raise ValueError(
                        "Stage-1 metric must be finite at row "
                        f"{row_number} in {csv_path}; "
                        f"got {score!r}."
                    )

                metric_rows.append(
                    {
                        "evaluation_round": round_number,
                        "evaluation_score": score,
                    }
                )
    except OSError as error:
        raise OSError(
            f"Could not read Stage-1 metrics CSV: {csv_path}"
        ) from error

    if not metric_rows:
        raise ValueError(
            f"Stage-1 metrics CSV contains no data rows: {csv_path}"
        )

    final_round = metric_rows[-1]["evaluation_round"]
    if final_round != expected_final_round:
        raise ValueError(
            "Stage-1 metrics CSV does not end at the expected "
            f"round {expected_final_round}: {csv_path} ends at "
            f"round {final_round}."
        )

    normalized_mode = str(evaluation_mode).strip().lower()
    if normalized_mode == "min":
        selected_metric = min(
            metric_rows,
            key=lambda row: row["evaluation_score"],
        )
    elif normalized_mode == "max":
        selected_metric = max(
            metric_rows,
            key=lambda row: row["evaluation_score"],
        )
    elif normalized_mode == "last_round":
        selected_metric = metric_rows[-1]
    else:
        raise ValueError(
            "evaluation.mode must be 'min', 'max', or "
            f"'last_round'; got {evaluation_mode!r}."
        )

    return dict(selected_metric)


def get_metric_selection_mode(
    metric,
    evaluation_mode,
):
    normalized_mode = str(evaluation_mode).strip().lower()
    if normalized_mode in {"min", "max"}:
        return normalized_mode

    if normalized_mode != "last_round":
        raise ValueError(
            "evaluation.mode must be 'min', 'max', or "
            f"'last_round'; got {evaluation_mode!r}."
        )

    normalized_metric = (
        str(metric).strip().lower().replace(" ", "_")
    )
    if normalized_metric.endswith("_loss"):
        return "min"
    if normalized_metric.endswith(("_accuracy", "_acc")):
        return "max"

    raise ValueError(
        "Cannot infer whether a last-round metric should be "
        f"minimized or maximized from {metric!r}."
    )


def record_stage_1_run_scores(
    stage_1_plan,
    config,
):
    metric = str(config.evaluation.metric)
    evaluation_mode = str(config.evaluation.mode)
    expected_final_round = (
        int(config.simulation.stage_1_end) - 1
    )
    if expected_final_round < 0:
        raise ValueError(
            "simulation.stage_1_end must be positive; "
            f"got {config.simulation.stage_1_end!r}."
        )

    simulations_root = (
        Path(config.output.results_root)
        / str(config.name)
        / str(config.run_id)
        / "simulations"
    )
    score_cache = {}

    for point in stage_1_plan["points"]:
        for trial in point["trials"]:
            for sampled_run in trial["sampled_stage_1_runs"]:
                run_directory = sampled_run[
                    "stage_1_run_directory"
                ]
                csv_path = (
                    simulations_root
                    / run_directory
                    / "stage_1.csv"
                )

                if run_directory not in score_cache:
                    score_cache[run_directory] = (
                        load_stage_1_metric(
                            csv_path=csv_path,
                            metric=metric,
                            evaluation_mode=evaluation_mode,
                            expected_final_round=(
                                expected_final_round
                            ),
                        )
                    )

                sampled_run.update(score_cache[run_directory])
                sampled_run["evaluation_metric"] = metric
                sampled_run["evaluation_mode"] = evaluation_mode
                sampled_run["stage_1_metrics_path"] = str(
                    csv_path
                )

    return stage_1_plan


def select_top_m_stage_1_runs(
    stage_1_plan,
    m,
    mode,
    seed,
):
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)):
        raise ValueError(
            f"top-m must be an integer; got {m!r}."
        )
    m = int(m)
    if m <= 0:
        raise ValueError(
            f"top-m must be positive; got {m!r}."
        )

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"min", "max"}:
        raise ValueError(
            "evaluation.mode must be either 'min' or 'max'; "
            f"got {mode!r}."
        )

    for point_index, point in enumerate(
        stage_1_plan["points"]
    ):
        for trial_index, trial in enumerate(point["trials"]):
            sampled_runs = trial["sampled_stage_1_runs"]
            if len(sampled_runs) < m:
                raise ValueError(
                    f"Cannot select top-{m} from only "
                    f"{len(sampled_runs)} runs at point "
                    f"{point_index}, trial {trial_index}."
                )

            trial_id = int(trial.get("trial", trial_index))
            tie_break_seed = [
                int(seed),
                point_index,
                trial_id,
            ]
            rng = np.random.default_rng(
                np.random.SeedSequence(tie_break_seed)
            )
            tie_break_values = rng.random(len(sampled_runs))

            if normalized_mode == "min":
                score_key = lambda index: (
                    sampled_runs[index]["evaluation_score"],
                    tie_break_values[index],
                )
            else:
                score_key = lambda index: (
                    -sampled_runs[index]["evaluation_score"],
                    tie_break_values[index],
                )

            ranked_indices = sorted(
                range(len(sampled_runs)),
                key=score_key,
            )

            top_m_runs = []
            for rank, sampled_index in enumerate(
                ranked_indices[:m],
                start=1,
            ):
                selected_run = dict(sampled_runs[sampled_index])
                selected_run["selection_rank"] = rank
                top_m_runs.append(selected_run)

            trial["selection_metric"] = str(
                sampled_runs[0]["evaluation_metric"]
            )
            trial["evaluation_mode"] = str(
                sampled_runs[0]["evaluation_mode"]
            )
            trial["selection_mode"] = normalized_mode
            trial["selection_tie_break_seed"] = tie_break_seed
            trial["top_m_stage_1_runs"] = top_m_runs

    return stage_1_plan


class Papernot_Baseline:
    def __init__(self, config):
        self.config = config

    def calculate_E_K_given_compute(self, compute, local_updates_schedule):
        E_K = compute/sum(local_updates_schedule)
        return E_K


class N_Stage_Method:
    def __init__(self, config):
        self.config = config.n_stage_tuning
    
    def calculate_compute_given_E_K(self, E_K, local_updates_schedule):
        E_K_each_stage = [E_K] + self.config.E_K_each_stage
        compute = 0
        for i, E_K in enumerate(E_K_each_stage):
            compute += E_K * local_updates_schedule[i]
        return compute
    
def utility_compute_plot(config: DictConfig) -> None:
    exp_config = config.experiment
    baseline = Papernot_Baseline(exp_config)
    n_stage_tuning = N_Stage_Method(exp_config)

    if exp_config.run_mode.generate_stage_1_plan:
        E_K_values_N_stage = exp_config.base_E_K_list
        local_updates_schedule = exp_config.local_updates_schedule
        fixed_compute = np.zeros_like(E_K_values_N_stage, dtype=float)
        E_K_values_papernot_baseline = np.zeros_like(E_K_values_N_stage, dtype=float)
        for i, E_K in enumerate(E_K_values_N_stage):
            fixed_compute[i] = n_stage_tuning.calculate_compute_given_E_K(E_K, local_updates_schedule)
            E_K_values_papernot_baseline[i] = baseline.calculate_E_K_given_compute(fixed_compute[i], local_updates_schedule)
        generate_plan(exp_config, 1, E_K_values_papernot_baseline, exp_config.num_trials, exp_config.run_id, exp_config.hp_configuration_ids, "papernot_baseline.JSON")
        generate_plan(exp_config, exp_config.n_stage_tuning.E_K_each_stage[0], E_K_values_N_stage, exp_config.num_trials, exp_config.run_id, exp_config.hp_configuration_ids, "two_stage_tuning_stage_1.JSON")


    if exp_config.run_mode.run_baseline_simulation:
        set_seed(config.run_settings.seed)
        if config.dataset.name == 'synthetic':
            model = getattr(models, config.dataset.model_name)(input_dim=config.dataset.dim_input, output_dim=config.dataset.dim_output)
        else:
            model = getattr(models, config.dataset.model_name)()
        learning_rate = get_selected_learning_rate(config)
        if config.server.constant_global_step == 'Fixed':
            global_step = config.server.global_step
            local_step = learning_rate
        elif config.server.constant_global_step == 'Adaptive':
            global_step = (config.server.client_ratio*config.dataset.nb_users)**0.5
            local_step = learning_rate/(config.server.local_updates*global_step)
        main_path = Path(HydraConfig.get().runtime.output_dir)/ "simulations"
        clipping_value = config.server.max_grad_norm
        for seed in range(exp_config.simulation.times):
            save_path = main_path / config.experiment.simulation.run_hp_configuration / f"seed_{seed}"
            save_path.mkdir(parents=True, exist_ok=True)
            train_data_loader, test_data_loader = get_data_loaders(config, per_client_loader=True)
            server = FedAvg(
                    model=model,
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
                    max_grad_norm=clipping_value, 
                    x_label=config.dataset.x_label,
                    y_label=config.dataset.y_label, 
                    client_sampling_scheme=config.server.client_sampling_scheme, 
                    data_sampling_scheme=config.server.data_sampling_scheme,
                    stage=exp_config.simulation.stage, 
                    stage_1_end=exp_config.simulation.stage_1_end, 
                    base_seed=seed
                )
            server.train()
            OmegaConf.save(config, save_path / f"stage_{exp_config.simulation.stage}_config.yaml", resolve=True)

    if exp_config.run_mode.generate_stage_2_plan:
        stage_1_plan = load_stage_1_plan(exp_config)
        stage_1_plan = map_stage_1_plan_runs(
            stage_1_plan,
            exp_config.hp_configuration_ids,
        )
        stage_1_plan = record_stage_1_run_scores(
            stage_1_plan,
            exp_config,
        )
        stage_1_plan = select_top_m_stage_1_runs(
            stage_1_plan,
            m=exp_config.n_stage_tuning.E_K_each_stage[0],
            mode=get_metric_selection_mode(
                metric=exp_config.evaluation.metric,
                evaluation_mode=exp_config.evaluation.mode,
            ),
            seed=exp_config.seed,
        )






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
