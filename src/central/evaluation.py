"""Held-out evaluation of explicitly selected central checkpoints."""

import math
from pathlib import Path

import torch

from central.data import get_data_loaders
from central.model import build_model


class HeldoutCheckpointEvaluator:
    """Evaluate each selected peak checkpoint once on held-out public data."""

    def __init__(self, config):
        self.config = config
        self.dataset_config = config.experiment.dataset
        _, self.validation_data_loader, self.heldout_data_loader = (
            get_data_loaders(
                config,
                seed=int(config.experiment.seed),
                split_public_evaluation=True,
            )
        )
        self.x_label = str(self.dataset_config.x_label)
        self.y_label = str(self.dataset_config.y_label)
        self.use_cuda = bool(config.run_settings.use_cuda)
        if self.use_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "use_cuda=True, but CUDA is not available for held-out "
                "checkpoint evaluation."
            )
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        try:
            loss_class = getattr(
                torch.nn,
                str(self.dataset_config.loss_fn),
            )
        except AttributeError as error:
            raise ValueError(
                "Unknown torch loss function "
                f"{self.dataset_config.loss_fn!r}."
            ) from error
        self.loss_fn = loss_class(reduction="mean")
        self._cache = {}
        self._fallback_cache = {}

    @staticmethod
    def _extract_batch(batch, x_label, y_label):
        if isinstance(batch, dict):
            return batch[x_label], batch[y_label]
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        raise ValueError(
            "Centralized data batches must be (features, labels) pairs "
            "or dictionaries containing the configured labels."
        )

    @staticmethod
    def _load_peak_checkpoint(checkpoint_path):
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as error:
            raise RuntimeError(
                f"Could not load selected peak checkpoint {checkpoint_path}."
            ) from error
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Peak checkpoint {checkpoint_path} must be a mapping."
            )
        if checkpoint.get("artifact_type") != "selected_peak_model":
            raise ValueError(
                f"Checkpoint {checkpoint_path} is not a selected peak "
                "model artifact."
            )
        if "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Peak checkpoint {checkpoint_path} has no model weights."
            )
        return checkpoint

    @staticmethod
    def _validate_selected_checkpoint(checkpoint, selected_run, path):
        selection = checkpoint.get("selection")
        selected_selection = selected_run.get("selection")
        if not isinstance(selection, dict) or not isinstance(
            selected_selection,
            dict,
        ):
            raise ValueError(
                f"Selected checkpoint metadata is incomplete: {path}."
            )
        exact_fields = ("metric", "mode", "stage", "round")
        for field in exact_fields:
            if selection.get(field) != selected_selection.get(field):
                raise ValueError(
                    f"Selected checkpoint {path} has {field}="
                    f"{selection.get(field)!r}, but compilation selected "
                    f"{selected_selection.get(field)!r}."
                )
        if not math.isclose(
            float(selection.get("score")),
            float(selected_selection.get("score")),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Selected checkpoint {path} score does not match the "
                "compiled validation score."
            )

    def _evaluate_model(self, model, data_loader, metric_prefix):
        model.to(self.device)
        model.eval()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for batch in data_loader:
                features, labels = self._extract_batch(
                    batch,
                    self.x_label,
                    self.y_label,
                )
                features = features.to(self.device)
                labels = labels.to(self.device)
                outputs = model(features)
                loss_values = self.loss_fn(outputs, labels)
                if not torch.isfinite(loss_values).all():
                    raise FloatingPointError(
                        "Held-out evaluation produced a non-finite loss."
                    )
                batch_size = labels.numel()
                total_loss += loss_values.mean().item() * batch_size
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_examples += batch_size
        if total_examples == 0:
            raise RuntimeError(
                f"Cannot evaluate an empty {metric_prefix} data loader."
            )
        return {
            f"{metric_prefix}_loss": total_loss / total_examples,
            f"{metric_prefix}_accuracy": total_correct / total_examples,
        }

    def __call__(self, selected_run):
        checkpoint_path = Path(str(selected_run["release_checkpoint_path"]))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "The selected peak checkpoint does not exist: "
                f"{checkpoint_path}"
            )
        cache_key = str(checkpoint_path.resolve())
        if cache_key in self._cache:
            return dict(self._cache[cache_key])

        checkpoint = self._load_peak_checkpoint(checkpoint_path)
        self._validate_selected_checkpoint(
            checkpoint,
            selected_run,
            checkpoint_path,
        )
        model = build_model(self.dataset_config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        utility = self._evaluate_model(
            model,
            self.heldout_data_loader,
            "test",
        )
        self._cache[cache_key] = dict(utility)
        return utility

    def build_data_independent_fallback(
        self,
        *,
        fallback_definition,
        evaluation,
    ):
        """Build and evaluate the fixed public Poisson ``K=0`` output."""
        if not isinstance(fallback_definition, dict):
            raise ValueError("fallback_definition must be a mapping.")
        if (
            fallback_definition.get("mechanism")
            != "fixed_random_initialization"
            or fallback_definition.get("depends_on_private_data") is not False
            or fallback_definition.get("num_private_training_runs") != 0
        ):
            raise ValueError(
                "The K=0 fallback must be a fixed data-independent random "
                "initialization with zero private training runs."
            )
        seed = fallback_definition.get("model_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(
                "The K=0 fallback model seed must be a non-negative integer."
            )
        selection_metric = str(evaluation["selection_metric"])
        if selection_metric not in {
            "validation_loss",
            "validation_accuracy",
        }:
            raise ValueError(
                "The K=0 fallback requires a public validation metric."
            )
        cache_key = (seed, selection_metric)
        if cache_key not in self._fallback_cache:
            # fork_rng restores the caller's Torch RNG state. The model is
            # built on CPU and never touches private training data.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = build_model(self.dataset_config)
            validation = self._evaluate_model(
                model,
                self.validation_data_loader,
                "validation",
            )
            utility = self._evaluate_model(
                model,
                self.heldout_data_loader,
                "test",
            )
            self._fallback_cache[cache_key] = {
                "hp_configuration_id": "__poisson_k_zero_fallback__",
                "stage_1_run_index": "",
                "selection": {
                    "metric": selection_metric,
                    "mode": str(evaluation["evaluation_mode"]),
                    "selection_mode": str(evaluation["selection_mode"]),
                    "stage": 0,
                    "round": -1,
                    "score": float(validation[selection_metric]),
                },
                "utility": {
                    metric: float(utility[metric])
                    for metric in evaluation["utility_metrics"]
                },
                "utility_source": (
                    "heldout_test_data_independent_fallback_evaluation"
                ),
                "candidate_origin": (
                    "poisson_k_zero_data_independent_fallback"
                ),
                "release_stage": 0,
                "release_round": -1,
                "release_model_definition": {
                    "architecture": "configured_central_model",
                    "initialization": "fixed_torch_seed",
                    "model_seed": seed,
                    "depends_on_private_data": False,
                },
            }
        return dict(self._fallback_cache[cache_key])
