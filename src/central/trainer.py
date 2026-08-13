"""Update-count-based centralized training with staged checkpoints."""

import copy
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.seed_utils import (
    DATA_SAMPLING_STREAM,
    DP_NOISE_STREAM,
    derive_seed,
)


_METRIC_HEADER = [
    "Round",
    "Train Loss",
    "Test Loss",
    "Train Accuracy",
    "Test Accuracy",
]


class CentralTrainer:
    """Train one centralized model for a fixed number of updates.

    A centralized ``round`` is exactly one completed optimizer update.
    Stage 2 is warm-started from the Stage-1 model, while its optimizer and
    privacy mechanism start afresh. Resuming an interrupted stage restores
    that stage's optimizer and privacy state.
    """

    def __init__(
        self,
        model,
        train_data_loader,
        test_data_loader,
        num_iters,
        save_path,
        loss_fn_name,
        learning_rate,
        weight_decay,
        use_cuda,
        dp,
        sample_rate,
        noise_multiplier,
        max_grad_norm,
        x_label="x",
        y_label="y",
        data_sampling_scheme="poisson_sampling",
        stage=None,
        stage_1_end=None,
        base_seed=0,
        stage_1_source_path=None,
        optimizer_name="SGD",
        momentum=0.0,
        checkpoint_interval=10,
        evaluation_interval=1,
    ):
        self.stage = self._validate_stage(stage)
        self.stage_1_end = self._validate_non_negative_integer(
            stage_1_end,
            "stage_1_end",
            allow_none=True,
        )
        self.num_iters = self._validate_non_negative_integer(
            num_iters,
            "num_iters",
        )
        self.base_seed = self._validate_non_negative_integer(
            base_seed,
            "base_seed",
        )
        self.checkpoint_interval = self._validate_positive_integer(
            checkpoint_interval,
            "checkpoint_interval",
        )
        self.evaluation_interval = self._validate_positive_integer(
            evaluation_interval,
            "evaluation_interval",
        )
        self._validate_stage_boundaries(stage_1_source_path)

        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.stage_1_source_path = (
            self.save_path
            if stage_1_source_path is None
            else Path(stage_1_source_path)
        )
        self.file_name = (
            f"stage_{self.stage}"
            if self.stage is not None
            else "metrics"
        )
        self.metrics_path = self.save_path / f"{self.file_name}.csv"
        self.checkpoint_path = self.save_path / f"{self.file_name}.pt"

        self.use_cuda = bool(use_cuda)
        if self.use_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "use_cuda=True, but CUDA is not available to PyTorch."
            )
        self.device = torch.device(
            "cuda"
            if self.use_cuda and torch.cuda.is_available()
            else "cpu"
        )
        self.dp = bool(dp)
        self.x_label = str(x_label)
        self.y_label = str(y_label)
        self.data_sampling_scheme = str(data_sampling_scheme)
        if self.data_sampling_scheme not in {
            "fixed_size_sampling",
            "poisson_sampling",
        }:
            raise ValueError(
                "data_sampling_scheme must be 'fixed_size_sampling' "
                f"or 'poisson_sampling'; got "
                f"{self.data_sampling_scheme!r}."
            )

        self.evaluation_train_data_loader = train_data_loader
        self.train_data_loader = self._build_training_loader(
            train_data_loader,
            sample_rate,
        )
        self.test_data_loader = test_data_loader
        if len(self.train_data_loader.dataset) == 0:
            raise ValueError("The centralized training dataset is empty.")
        if len(self.test_data_loader.dataset) == 0:
            raise ValueError("The centralized test dataset is empty.")

        self.model = copy.deepcopy(model)
        self.start_iter = 0
        self.resume_from_checkpoint = False
        self.initialized_from_stage_1 = False
        checkpoint = self._initialize_model()
        self.model.to(self.device)

        try:
            loss_class = getattr(torch.nn, str(loss_fn_name))
        except AttributeError as error:
            raise ValueError(
                f"Unknown torch loss function {loss_fn_name!r}."
            ) from error
        self.loss_fn = loss_class(reduction="mean")

        try:
            optimizer_class = getattr(torch.optim, str(optimizer_name))
        except AttributeError as error:
            raise ValueError(
                f"Unknown torch optimizer {optimizer_name!r}."
            ) from error
        optimizer_kwargs = {
            "lr": self._validate_positive_float(
                learning_rate,
                "learning_rate",
            ),
            "weight_decay": self._validate_non_negative_float(
                weight_decay,
                "weight_decay",
            ),
        }
        if str(optimizer_name) == "SGD":
            optimizer_kwargs["momentum"] = self._validate_non_negative_float(
                momentum,
                "momentum",
            )
        self.optimizer = optimizer_class(
            self.model.parameters(),
            **optimizer_kwargs,
        )

        self.privacy_engine = None
        self.noise_generator = None
        self.private_loss_fn = None
        if self.dp:
            self._make_private(
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
            if self.data_sampling_scheme == "poisson_sampling":
                self._validate_private_sample_rate(sample_rate)

        if self.resume_from_checkpoint:
            self._restore_training_state(checkpoint)
        self._initialize_metrics_file()

    @staticmethod
    def _validate_stage(stage):
        if stage is None:
            return None
        if isinstance(stage, bool) or not isinstance(stage, int):
            raise ValueError(f"stage must be 1 or 2; got {stage!r}.")
        if stage not in {1, 2}:
            raise ValueError(f"stage must be 1 or 2; got {stage!r}.")
        return stage

    @staticmethod
    def _validate_non_negative_integer(value, name, allow_none=False):
        if allow_none and value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"{name} must be a non-negative integer; got {value!r}."
            )
        return value

    @classmethod
    def _validate_positive_integer(cls, value, name):
        value = cls._validate_non_negative_integer(value, name)
        if value == 0:
            raise ValueError(
                f"{name} must be a positive integer; got {value!r}."
            )
        return value

    @staticmethod
    def _validate_positive_float(value, name):
        parsed_value = float(value)
        if not torch.isfinite(torch.tensor(parsed_value)) or parsed_value <= 0:
            raise ValueError(
                f"{name} must be finite and positive; got {value!r}."
            )
        return parsed_value

    @staticmethod
    def _validate_non_negative_float(value, name):
        parsed_value = float(value)
        if not torch.isfinite(torch.tensor(parsed_value)) or parsed_value < 0:
            raise ValueError(
                f"{name} must be finite and non-negative; got {value!r}."
            )
        return parsed_value

    def _validate_stage_boundaries(self, stage_1_source_path):
        if stage_1_source_path is not None and self.stage != 2:
            raise ValueError(
                "stage_1_source_path may only be provided for Stage 2."
            )
        if self.stage == 1:
            if self.stage_1_end is None:
                raise ValueError("stage_1_end is required for Stage 1.")
            if self.num_iters != self.stage_1_end:
                raise ValueError(
                    "For Stage 1, num_iters must equal stage_1_end; "
                    f"got {self.num_iters} and {self.stage_1_end}."
                )
        if self.stage == 2:
            if self.stage_1_end is None:
                raise ValueError("stage_1_end is required for Stage 2.")
            if stage_1_source_path is None:
                raise ValueError(
                    "stage_1_source_path is required for Stage 2."
                )
            if self.num_iters <= self.stage_1_end:
                raise ValueError(
                    "For Stage 2, num_iters must be greater than "
                    f"stage_1_end ({self.stage_1_end}); got "
                    f"{self.num_iters}."
                )

    def _build_training_loader(self, train_data_loader, sample_rate):
        sample_rate = float(sample_rate)
        if not 0 < sample_rate <= 1:
            raise ValueError(
                "sample_rate must be in (0, 1]; "
                f"got {sample_rate!r}."
            )
        dataset = train_data_loader.dataset
        batch_size = max(1, int(sample_rate * len(dataset)))
        generator = torch.Generator().manual_seed(self.base_seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            generator=generator,
        )

    @staticmethod
    def _load_checkpoint(checkpoint_path):
        try:
            # These are trusted, locally generated training checkpoints and
            # include optimizer, RNG, and Opacus accountant state in addition
            # to model tensors, so PyTorch's weights-only loader is not
            # sufficient.
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as error:
            raise RuntimeError(
                f"Could not load checkpoint {checkpoint_path!s}."
            ) from error
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Checkpoint {checkpoint_path!s} must contain a mapping."
            )
        if "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint {checkpoint_path!s} does not contain "
                "'model_state_dict'."
            )
        return checkpoint

    @staticmethod
    def _completed_iters(checkpoint, checkpoint_path):
        completed_iters = checkpoint.get("rounds")
        if completed_iters is None and "round" in checkpoint:
            completed_iters = checkpoint["round"] + 1
        if (
            isinstance(completed_iters, bool)
            or not isinstance(completed_iters, int)
            or completed_iters < 0
        ):
            raise ValueError(
                f"Checkpoint {checkpoint_path!s} has invalid completed "
                f"update count {completed_iters!r}."
            )
        return completed_iters

    def _initialize_model(self):
        if self.checkpoint_path.is_file():
            checkpoint = self._load_checkpoint(self.checkpoint_path)
            checkpoint_stage = checkpoint.get("stage")
            if checkpoint_stage != self.stage:
                raise RuntimeError(
                    f"Checkpoint {self.checkpoint_path!s} belongs to "
                    f"stage {checkpoint_stage!r}, not stage "
                    f"{self.stage!r}."
                )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.start_iter = self._completed_iters(
                checkpoint,
                self.checkpoint_path,
            )
            if self.start_iter > self.num_iters:
                raise RuntimeError(
                    f"Checkpoint records {self.start_iter} completed "
                    f"updates, but num_iters is only {self.num_iters}."
                )
            if self.stage == 2 and self.start_iter < self.stage_1_end:
                raise RuntimeError(
                    "The Stage-2 checkpoint predates the Stage-1 "
                    "boundary."
                )
            self.resume_from_checkpoint = True
            return checkpoint

        if self.stage != 2:
            return None

        stage_1_path = self.stage_1_source_path / "stage_1.pt"
        if not stage_1_path.is_file():
            raise FileNotFoundError(
                "Cannot start Stage 2 because the Stage-1 checkpoint "
                f"does not exist: {stage_1_path!s}"
            )
        stage_1_checkpoint = self._load_checkpoint(stage_1_path)
        if stage_1_checkpoint.get("stage") != 1:
            raise RuntimeError(
                f"Checkpoint {stage_1_path!s} is not a Stage-1 "
                "checkpoint."
            )
        completed_iters = self._completed_iters(
            stage_1_checkpoint,
            stage_1_path,
        )
        if completed_iters != self.stage_1_end:
            raise RuntimeError(
                "Cannot start Stage 2 because Stage 1 is incomplete: "
                f"checkpoint records {completed_iters} updates, "
                f"expected {self.stage_1_end}."
            )
        self._validate_metrics_file(
            metrics_path=self.stage_1_source_path / "stage_1.csv",
            completed_iters=self.stage_1_end,
            start_round=0,
        )
        self.model.load_state_dict(stage_1_checkpoint["model_state_dict"])
        self.start_iter = self.stage_1_end
        self.initialized_from_stage_1 = True
        return None

    def _make_private(self, noise_multiplier, max_grad_norm):
        try:
            from opacus import PrivacyEngine
        except ImportError as error:
            raise ImportError(
                "Central DP training requires the 'opacus' package."
            ) from error

        self.noise_generator = torch.Generator(
            device=self.device.type
        ).manual_seed(
            derive_seed(
                self.base_seed,
                self.stage or 0,
                stream=DP_NOISE_STREAM,
            )
        )
        self.privacy_engine = PrivacyEngine()
        private_objects = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_data_loader,
            noise_multiplier=self._validate_positive_float(
                noise_multiplier,
                "noise_multiplier",
            ),
            max_grad_norm=self._validate_positive_float(
                max_grad_norm,
                "max_grad_norm",
            ),
            noise_generator=self.noise_generator,
            criterion=self.loss_fn,
            grad_sample_mode="ghost",
            loss_reduction="mean",
            poisson_sampling=(
                self.data_sampling_scheme == "poisson_sampling"
            ),
        )
        if len(private_objects) == 4:
            (
                self.model,
                self.optimizer,
                self.private_loss_fn,
                self.train_data_loader,
            ) = private_objects
        else:
            (
                self.model,
                self.optimizer,
                self.train_data_loader,
            ) = private_objects

    def _validate_private_sample_rate(self, configured_sample_rate):
        actual_sample_rate = getattr(
            self.train_data_loader,
            "sample_rate",
            None,
        )
        if actual_sample_rate is None:
            actual_sample_rate = getattr(
                getattr(self.train_data_loader, "batch_sampler", None),
                "sample_rate",
                None,
            )
        if actual_sample_rate is None:
            raise RuntimeError(
                "Could not determine the effective Opacus Poisson "
                "sampling rate."
            )
        if not math.isclose(
            float(actual_sample_rate),
            float(configured_sample_rate),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "The configured DP-SGD sampling rate does not match "
                "the effective Opacus rate: configured="
                f"{float(configured_sample_rate)}, effective="
                f"{float(actual_sample_rate)}. Choose a rate compatible "
                "with the dataset size so training and accounting match."
            )

    def _restore_training_state(self, checkpoint):
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is None:
            raise ValueError(
                f"Checkpoint {self.checkpoint_path!s} does not contain "
                "'optimizer_state_dict'."
            )
        self.optimizer.load_state_dict(optimizer_state)

        if self.dp:
            noise_state = checkpoint.get("noise_generator_state")
            accountant_state = checkpoint.get("accountant_state_dict")
            if noise_state is None or accountant_state is None:
                raise ValueError(
                    "A private checkpoint must contain noise-generator "
                    "and privacy-accountant state."
                )
            self.noise_generator.set_state(noise_state)
            self.privacy_engine.accountant.load_state_dict(
                accountant_state
            )

    def _initialize_metrics_file(self):
        start_round = self.stage_1_end if self.stage == 2 else 0
        if self.resume_from_checkpoint:
            self._validate_and_truncate_metrics(
                completed_iters=self.start_iter,
                start_round=start_round,
            )
            return
        with self.metrics_path.open(
            mode="w",
            encoding="utf-8",
            newline="",
        ) as file:
            csv.writer(file).writerow(_METRIC_HEADER)

    def _validate_and_truncate_metrics(
        self,
        completed_iters,
        start_round,
    ):
        rows = self._validate_metrics_file(
            metrics_path=self.metrics_path,
            completed_iters=completed_iters,
            start_round=start_round,
        )
        retained_rows = [
            row
            for row in rows[1:]
            if row and int(row[0]) < completed_iters
        ]
        with self.metrics_path.open(
            mode="w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(_METRIC_HEADER)
            writer.writerows(retained_rows)

    def _validate_metrics_file(
        self,
        metrics_path,
        completed_iters,
        start_round,
    ):
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"Metrics CSV does not exist: {metrics_path}"
            )
        with metrics_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as file:
            rows = list(csv.reader(file))
        if not rows or rows[0] != _METRIC_HEADER:
            raise RuntimeError(
                f"Metrics CSV {metrics_path} has an invalid header."
            )
        expected_rounds = [
            iteration
            for iteration in range(start_round, completed_iters)
            if (iteration + 1) % self.evaluation_interval == 0
        ]
        final_iteration = completed_iters - 1
        if (
            completed_iters == self.num_iters
            and completed_iters > start_round
            and final_iteration not in expected_rounds
        ):
            expected_rounds.append(final_iteration)
        try:
            observed_rounds = [
                int(row[0])
                for row in rows[1:]
                if row
            ]
        except (IndexError, ValueError) as error:
            raise RuntimeError(
                f"Metrics CSV {metrics_path} contains an invalid round."
            ) from error
        observed_completed_rounds = [
            iteration
            for iteration in observed_rounds
            if iteration < completed_iters
        ]
        if observed_completed_rounds != expected_rounds:
            raise RuntimeError(
                f"Metrics CSV {metrics_path} does not contain the "
                "expected evaluation rounds "
                f"{expected_rounds!r} through update "
                f"{completed_iters - 1}."
            )
        return rows

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

    def _evaluate_loader(self, data_loader):
        self.model.eval()
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
                outputs = self.model(features)
                batch_size = labels.numel()
                # Opacus' ghost-clipping setup changes the wrapped
                # criterion to return one loss per example. Evaluation
                # must aggregate that vector back to the configured mean.
                loss_values = self.loss_fn(outputs, labels)
                if not torch.isfinite(loss_values).all():
                    raise FloatingPointError(
                        "Evaluation produced a non-finite loss."
                    )
                batch_loss = loss_values.mean().item()
                total_loss += batch_loss * batch_size
                total_correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()
                total_examples += batch_size
        if total_examples == 0:
            raise RuntimeError("Cannot evaluate an empty data loader.")
        return (
            total_loss / total_examples,
            total_correct / total_examples,
        )

    def evaluate(self, iteration):
        train_loss, train_accuracy = self._evaluate_loader(
            self.evaluation_train_data_loader
        )
        test_loss, test_accuracy = self._evaluate_loader(
            self.test_data_loader
        )
        with self.metrics_path.open(
            mode="a",
            encoding="utf-8",
            newline="",
        ) as file:
            csv.writer(file).writerow(
                [
                    iteration,
                    train_loss,
                    test_loss,
                    train_accuracy,
                    test_accuracy,
                ]
            )
        return {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }

    def _base_model(self):
        return getattr(self.model, "_module", self.model)

    def save_checkpoint(self, completed_iters):
        checkpoint = {
            "stage": self.stage,
            "rounds": completed_iters,
            "round": completed_iters - 1,
            "model_state_dict": self._base_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.dp:
            checkpoint["noise_generator_state"] = (
                self.noise_generator.get_state()
            )
            checkpoint["accountant_state_dict"] = (
                self.privacy_engine.accountant.state_dict()
            )
        temporary_path = self.checkpoint_path.with_suffix(".pt.tmp")
        try:
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(self.checkpoint_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _get_iteration_batch(self, iteration):
        sampling_seed = derive_seed(
            self.base_seed,
            self.stage or 0,
            iteration,
            stream=DATA_SAMPLING_STREAM,
        )
        generator = torch.Generator().manual_seed(sampling_seed)
        if self.dp:
            try:
                from opacus.data_loader import switch_generator
            except ImportError as error:
                raise ImportError(
                    "Central DP training requires the 'opacus' package."
                ) from error
            self.train_data_loader = switch_generator(
                data_loader=self.train_data_loader,
                generator=generator,
            )
        else:
            self.train_data_loader.generator = generator
            sampler = getattr(self.train_data_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "generator"):
                sampler.generator = generator
        return next(iter(self.train_data_loader))

    def train(self):
        for iteration in range(self.start_iter, self.num_iters):
            batch = self._get_iteration_batch(iteration)

            features, labels = self._extract_batch(
                batch,
                self.x_label,
                self.y_label,
            )
            if labels.numel() == 0:
                raise RuntimeError(
                    "The sampled central batch is empty. Empty Poisson "
                    "batches are not yet supported by CentralTrainer."
                )
            features = features.to(self.device)
            labels = labels.to(self.device)
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss_function = (
                self.private_loss_fn
                if self.private_loss_fn is not None
                else self.loss_fn
            )
            loss = loss_function(outputs, labels)
            finite_check_loss = getattr(
                loss,
                "loss_per_sample",
                loss,
            )
            if not torch.isfinite(finite_check_loss).all():
                raise FloatingPointError(
                    "Training produced a non-finite loss at optimizer "
                    f"update {iteration}."
                )
            loss.backward()
            self.optimizer.step()

            completed_iters = iteration + 1
            should_evaluate = (
                completed_iters % self.evaluation_interval == 0
                or completed_iters == self.num_iters
            )
            if should_evaluate:
                metrics = self.evaluate(iteration)
                stage_label = (
                    f"Stage {self.stage}"
                    if self.stage is not None
                    else "Central training"
                )
                print(
                    f"{stage_label}: completed {completed_iters}/"
                    f"{self.num_iters} updates; test_loss="
                    f"{metrics['test_loss']:.6f}; test_accuracy="
                    f"{metrics['test_accuracy']:.6f}",
                    flush=True,
                )
            if completed_iters % self.checkpoint_interval == 0:
                self.save_checkpoint(completed_iters)

        self.save_checkpoint(self.num_iters)
        return self.checkpoint_path
