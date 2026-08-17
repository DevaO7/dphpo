import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from hpo import planning, results
from hpo.execution import get_selected_learning_rate
import tuning_experiments_central as central


def _config(results_root):
    return OmegaConf.create(
        {
            "run_settings": {
                "rounds": 2,
                "weight_decay": 0.0,
                "use_cuda": False,
                "dp": True,
                "sampling_rate": 0.01,
                "noise_multiplier": 5.0,
                "max_grad_norm": 1.0,
                "data_sampling_scheme": "poisson_sampling",
                "optimizer_name": "SGD",
                "momentum": 0.0,
                "checkpoint_interval": 1,
                "evaluation_interval": 1,
            },
            "hp_candidate_set": {
                "H1": {"step_size": 0.01},
                "H2": {"step_size": 0.02},
                "H3": {"step_size": 0.03},
                "H4": {"step_size": 0.04},
            },
            "experiment": {
                "name": "privacy_stage_2_test",
                "seed": 7,
                "run_id": "run_id_test",
                "eta": 0.0,
                "num_trials": 3,
                "hp_configuration_ids": ["H1", "H2", "H3", "H4"],
                "simulation": {
                    "method": "two_stage_tuning",
                    "run_hp_configuration": "H1",
                    "stage": 1,
                    "stage_1_end": 2,
                    "stage_2_end": 5,
                    "mu": 4,
                    "target_epsilon": 0.5,
                },
                "two_stage": {
                    "num_survivors": 3,
                    "stage_2_expected_trials": 2,
                },
                "privacy": {
                    "delta": 1e-5,
                    "max_renyi_order": 100,
                    "sigma_search": {
                        "initial_sigma": 5.0,
                        "minimum_sigma": 0.01,
                        "maximum_sigma": 100.0,
                        "relative_tolerance": 1e-6,
                        "max_iterations": 80,
                    },
                },
                "evaluation": {
                    "selection": {
                        "metric": "train_loss",
                        "mode": "min",
                    },
                    "utility": {
                        "metrics": ["train_loss"],
                        "at": "selection_round",
                    },
                },
                "dataset": {
                    "name": "synthetic",
                    "loss_fn": "CrossEntropyLoss",
                    "x_label": "x",
                    "y_label": "y",
                },
                "output": {"results_root": str(results_root)},
            },
        }
    )


def _calibration():
    return {
        "method": "two_stage_tuning",
        "target_epsilon": 0.5,
        "achieved_epsilon": 0.499999,
        "delta": 1e-5,
        "noise_multiplier": 7.5,
        "best_renyi_order": 32.0,
        "min_renyi_order": 2,
        "max_renyi_order": 100,
        "relative_sigma_tolerance": 1e-6,
        "bisection_iterations": 20,
        "accountant_evaluations": 22,
        "accounting_method": "numerical",
    }


class PrivacyMatchedStage2PlanningTest(unittest.TestCase):
    def test_stage_2_plan_uses_nested_metrics_and_preserves_privacy(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(temporary_directory)
            with mock.patch.object(
                planning,
                "_calibrate_noise_multiplier",
                return_value=_calibration(),
            ):
                planning.generate_privacy_matched_plan(
                    config=config,
                    method="two_stage_tuning",
                    target_epsilon=0.5,
                    E_k=4,
                )

            stage_1_plan = (
                planning.load_privacy_matched_simulation_plan(
                    config,
                    stage=1,
                )
            )
            simulations_root = (
                planning.get_privacy_matched_simulations_directory(
                    config=config,
                    method="two_stage_tuning",
                    target_epsilon=0.5,
                    E_k=4,
                )
            )
            for spec in stage_1_plan["execution_summary"][
                "required_stage_1_run_specs"
            ]:
                run_directory = (
                    simulations_root
                    / spec["stage_1_run_directory"]
                )
                run_directory.mkdir(parents=True, exist_ok=True)
                hp_number = int(spec["hp_configuration_id"][1:])
                run_index = int(spec["stage_1_run_index"])
                final_loss = hp_number + run_index / 100.0
                (run_directory / "stage_1.csv").write_text(
                    "Round,Train Loss\n"
                    f"0,{final_loss + 0.1}\n"
                    f"1,{final_loss}\n",
                    encoding="utf-8",
                )

            plan_path = (
                results.generate_privacy_matched_stage_2_plan_from_results(
                    config,
                    evaluation_interval=1,
                )
            )
            first_plan_text = Path(plan_path).read_text(
                encoding="utf-8"
            )
            repeated_plan_path = (
                results.generate_privacy_matched_stage_2_plan_from_results(
                    config,
                    evaluation_interval=1,
                )
            )
            self.assertEqual(plan_path, repeated_plan_path)
            self.assertEqual(
                first_plan_text,
                Path(repeated_plan_path).read_text(encoding="utf-8"),
            )

            plan = json.loads(first_plan_text)
            loaded_plan = (
                planning.load_privacy_matched_simulation_plan(
                    config,
                    stage=2,
                )
            )
            self.assertEqual(loaded_plan, plan)
            self.assertTrue(
                str(plan_path).endswith(
                    str(
                        Path(
                            "plan",
                            "two_stage_tuning",
                            "epsilon_0p5",
                            "mu_4",
                            "two_stage_tuning_stage_2.JSON",
                        )
                    )
                )
            )
            self.assertEqual(plan["plan_type"], "privacy_matched")
            self.assertEqual(len(plan["points"]), 1)
            point = plan["points"][0]
            self.assertEqual(point["target_epsilon"], 0.5)
            self.assertEqual(point["achieved_epsilon"], 0.499999)
            self.assertEqual(point["noise_multiplier"], 7.5)
            self.assertEqual(
                point["privacy_calibration"],
                _calibration(),
            )

            for trial in point["trials"]:
                survivors = trial["trial_stage_1"][
                    "top_m_stage_1_runs"
                ]
                self.assertEqual(len(survivors), 3)
                self.assertEqual(
                    [run["selection_rank"] for run in survivors],
                    [1, 2, 3],
                )
                self.assertEqual(
                    len(
                        {
                            run["stage_1_run_directory"]
                            for run in survivors
                        }
                    ),
                    3,
                )
                for run in survivors:
                    self.assertTrue(
                        str(run["stage_1_metrics_path"]).startswith(
                            str(simulations_root)
                        )
                    )

            required_specs = plan["execution_summary"][
                "required_stage_2_run_specs"
            ]
            self.assertTrue(required_specs)
            for spec in required_specs:
                self.assertTrue(
                    (
                        simulations_root
                        / spec["stage_1_run_directory"]
                        / "stage_1.csv"
                    ).is_file()
                )

            stage_1_execution_config = (
                central._config_with_plan_noise_multiplier(
                    config,
                    stage_1_plan,
                )
            )
            for spec in stage_1_plan["execution_summary"][
                "required_stage_1_run_specs"
            ]:
                stage_1_execution_config.experiment.simulation[
                    "run_hp_configuration"
                ] = spec["hp_configuration_id"]
                signature = central._training_signature(
                    config=stage_1_execution_config,
                    stage=1,
                    run_spec=spec,
                    learning_rate=get_selected_learning_rate(
                        stage_1_execution_config
                    ),
                )
                source_directory = (
                    simulations_root
                    / spec["stage_1_run_directory"]
                )
                (source_directory / "stage_1_training_signature.JSON").write_text(
                    json.dumps(signature),
                    encoding="utf-8",
                )
                (source_directory / "stage_1.pt").write_bytes(b"test")

            selected_hp_id = required_specs[0][
                "hp_configuration_id"
            ]
            config.experiment.simulation.stage = 2
            config.experiment.simulation.run_hp_configuration = (
                selected_hp_id
            )
            config.run_settings.rounds = 5
            trainer = mock.Mock()
            with mock.patch.object(
                central,
                "set_global_seed",
            ), mock.patch.object(
                central,
                "get_data_loaders",
                return_value=(object(), object()),
            ), mock.patch.object(
                central,
                "build_model",
                return_value=object(),
            ), mock.patch.object(
                central,
                "CentralTrainer",
                return_value=trainer,
            ) as trainer_class:
                central.run_planned_simulations(
                    config,
                    stage=2,
                    privacy_matched=True,
                )

            selected_specs = [
                spec
                for spec in required_specs
                if spec["hp_configuration_id"] == selected_hp_id
            ]
            self.assertEqual(
                trainer_class.call_count,
                len(selected_specs),
            )
            for trainer_call, spec in zip(
                trainer_class.call_args_list,
                selected_specs,
            ):
                self.assertEqual(
                    trainer_call.kwargs["noise_multiplier"],
                    7.5,
                )
                self.assertEqual(
                    trainer_call.kwargs["stage_1_source_path"],
                    simulations_root
                    / spec["stage_1_run_directory"],
                )
                continuation_directory = (
                    simulations_root
                    / spec["stage_2_run_directory"]
                )
                saved_config = OmegaConf.load(
                    continuation_directory
                    / "stage_2_config.yaml"
                )
                self.assertEqual(
                    saved_config.run_settings.noise_multiplier,
                    7.5,
                )


if __name__ == "__main__":
    unittest.main()
