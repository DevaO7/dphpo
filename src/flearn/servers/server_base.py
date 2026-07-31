import numpy as np
import os 
import csv
import copy
import matplotlib.pyplot as plt
import torch


class Server:
    def __init__(
        self,
        model,
        similarity,
        save_path,
        file_name,
        client_ratio,
        dp,
        use_cuda,
        num_glob_iters,
        client_sampling_scheme,
        data_sampling_scheme,
        stage=None,
        stage_1_end=None,
        base_seed=0,
        stage_1_source_path=None,
    ):
        self.users = []
        self.selected_users = []
        self.use_cuda = use_cuda
        self.save_path = save_path
        self.stage = self._validate_stage(stage)
        self.stage_1_end = self._validate_stage_1_end(stage_1_end)
        if stage_1_source_path is not None and self.stage != 2:
            raise ValueError(
                "stage_1_source_path may only be provided for "
                "Stage-2 runs."
            )
        self.stage_1_source_path = (
            save_path
            if stage_1_source_path is None
            else stage_1_source_path
        )
        self.model = copy.deepcopy(model)
        self.checkpoint = None
        self.resume_from_checkpoint = False
        self.initialized_from_stage_1 = False
        self.base_seed = base_seed

        if self.stage is None:
            # Preserve the pre-staging checkpoint location for existing callers.
            self.checkpoint_path = os.path.join(
                "checkpoints",
                save_path,
                "checkpoint.pth",
            )
            self._load_legacy_checkpoint_if_present()
        else:
            self.checkpoint_path = os.path.join(
                save_path,
                f"stage_{self.stage}.pt",
            )
            self._initialize_staged_model()

        self.similarity = similarity
        if self.stage is not None:
            self.file_name = f"stage_{self.stage}"
        elif file_name is None:
            self.file_name = "metrics"
        else:
            self.file_name = file_name
        self.client_ratio = client_ratio
        self.dp = dp
        self.num_glob_iters = num_glob_iters
        self.data_sampling_scheme = data_sampling_scheme
        self.client_sampling_scheme = client_sampling_scheme

    @staticmethod
    def _validate_stage(stage):
        if stage is None:
            return None
        if isinstance(stage, bool) or not isinstance(stage, int):
            raise ValueError(
                f"stage must be 1 or 2; got {stage!r}."
            )
        if stage not in (1, 2):
            raise ValueError(
                f"stage must be 1 or 2; got {stage!r}."
            )
        return stage

    @staticmethod
    def _validate_stage_1_end(stage_1_end):
        if stage_1_end is None:
            return None
        if (
            isinstance(stage_1_end, bool)
            or not isinstance(stage_1_end, int)
            or stage_1_end < 0
        ):
            raise ValueError(
                "stage_1_end must be a non-negative integer; "
                f"got {stage_1_end!r}."
            )
        return stage_1_end

    @staticmethod
    def _completed_rounds(checkpoint, checkpoint_path):
        if "rounds" in checkpoint:
            rounds = checkpoint["rounds"]
        elif "round" in checkpoint:
            # Compatibility with the old zero-based checkpoint schema.
            rounds = checkpoint["round"] + 1
        else:
            raise ValueError(
                f"Checkpoint {checkpoint_path!s} does not contain "
                "'rounds'."
            )

        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0:
            raise ValueError(
                f"Checkpoint {checkpoint_path!s} has invalid completed-round "
                f"count {rounds!r}."
            )
        return rounds

    @staticmethod
    def _load_checkpoint(checkpoint_path):
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load checkpoint {checkpoint_path!s}."
            ) from error
        if "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint {checkpoint_path!s} does not contain "
                "'model_state_dict'."
            )
        return checkpoint

    def _resume_from(self, checkpoint_path):
        self.checkpoint = self._load_checkpoint(checkpoint_path)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.start_iter = self._completed_rounds(
            self.checkpoint,
            checkpoint_path,
        )
        self.resume_from_checkpoint = True

    def _load_legacy_checkpoint_if_present(self):
        if os.path.exists(self.checkpoint_path):
            self._resume_from(self.checkpoint_path)
        else:
            self.start_iter = 0

    def _initialize_staged_model(self):
        if self.stage == 1:
            if os.path.exists(self.checkpoint_path):
                self._resume_from(self.checkpoint_path)
            else:
                self.start_iter = 0
            return

        if self.stage_1_end is None:
            raise ValueError(
                "stage_1_end is required when stage is 2."
            )

        stage_1_path = os.path.join(
            self.stage_1_source_path,
            "stage_1.pt",
        )
        if not os.path.exists(stage_1_path):
            raise FileNotFoundError(
                "Cannot start Stage 2 because the Stage 1 checkpoint "
                f"does not exist: {stage_1_path!s}"
            )

        stage_1_checkpoint = self._load_checkpoint(stage_1_path)
        stage_1_rounds = self._completed_rounds(
            stage_1_checkpoint,
            stage_1_path,
        )
        if stage_1_rounds != self.stage_1_end:
            raise RuntimeError(
                "Cannot start Stage 2 because Stage 1 is incomplete: "
                f"{stage_1_path!s} records {stage_1_rounds} completed "
                f"rounds, expected {self.stage_1_end}."
            )

        if os.path.exists(self.checkpoint_path):
            self._resume_from(self.checkpoint_path)
            if self.start_iter < self.stage_1_end:
                raise RuntimeError(
                    f"Stage 2 checkpoint {self.checkpoint_path!s} records "
                    f"{self.start_iter} completed rounds, which is before "
                    f"the Stage 1 boundary {self.stage_1_end}."
                )
            return

        # A new Stage 2 run is warm-started from the Stage 1 model only.
        # Per-user optimizer and noise-generator state is initialized afresh.
        self.model.load_state_dict(
            stage_1_checkpoint["model_state_dict"]
        )
        self.start_iter = self.stage_1_end
        self.initialized_from_stage_1 = True

    def send_parameters(self):
        """Users setting their parameters from the server."""
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)
    
    def save_checkpoint(self, completed_rounds):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        privacy_engine_generator = {}
        optimizer_state_dict = {}
        if self.dp:
            for user in self.users:
                privacy_engine_generator[user.id] = user.generator.get_state()
                optimizer_state_dict[user.id] = user.optimizer.state_dict()
        check_point = {
            "stage": self.stage,
            "rounds": completed_rounds,
            # Retain the old key so older analysis code can still read the
            # zero-based index of the last completed round.
            "round": completed_rounds - 1,
            'model_state_dict': self.model.state_dict(),
            'privacy_engine_generator': privacy_engine_generator,
            'optimizer_state_dict': optimizer_state_dict
        }
        torch.save(check_point, self.checkpoint_path)

    def poisson_sampling(self, data, probabilities, seed):
        """
        data: list or array of items
        probabilities: list or array of p_i for each item
        """
        # Generate independent random floats [0.0, 1.0) for each element
        rng = np.random.default_rng(seed)
        random_vals = rng.random(len(data))

        # Select indices where the random value is less than the assigned probability
        selected_mask = random_vals < probabilities
        return np.array(data)[selected_mask]

    def select_users_poisson_sampling(self, glob_iter):
        assert 0.0 < self.client_ratio <= 1.0
        ids = [c.id for c in self.users]
        probs = np.ones(len(self.users))*self.client_ratio
        selected_ids = self.poisson_sampling(ids, probs, seed=300*self.base_seed+glob_iter)
        print(f"Selected users: {selected_ids}")
        selected_set = set(map(int, selected_ids.tolist()))
        self.selected_users = [c for c in self.users if c.id in selected_set]
        return self.selected_users

    def select_users_fixed_sampling(self, glob_iter):
        assert 0.0 < self.client_ratio <= 1.0
        ids = [c.id for c in self.users]
        np.random.seed(self.base_seed*200+glob_iter)
        selected_set = np.random.choice(ids, size=max(1, int(self.client_ratio * len(self.users))), replace=False)
        print(f"Selected users: {selected_set}")
        self.selected_users = [c for c in self.users if c.id in selected_set]
        return self.selected_users
    
    def test_error_and_loss(self):
        """Excess error of the current model of all users (test data)"""
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.test_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def train_error_and_loss(self):
        """Excess error of the current model of all users (train data)"""
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.users]

        return ids, num_samples, tot_correct, losses

    def evaluate(self, glob_iter):
        """Saves the metrics at the beginning of each communication round."""
        stats_test = self.test_error_and_loss() # [0] ids, [1] num_samples, [2] tot_correct, [3] losses
        stats_train = self.train_error_and_loss() # [0] ids, [1] num_samples, [2] tot_correct, [3] losses
        glob_acc = np.sum(stats_test[2]) * 1.0 / np.sum(stats_test[1]) # Total correct / Total samples
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1]) # Total correct / Total samples
        test_loss  = np.sum(np.array(stats_test[3])  * np.array(stats_test[1]))  / np.sum(stats_test[1]) # Weighted avg loss
        train_loss = np.sum(np.array(stats_train[3]) * np.array(stats_train[1])) / np.sum(stats_train[1]) # Weighted avg loss
        print("Similarity:", self.similarity)
        print("Average Global Test Accuracy: ", round(glob_acc, 5))
        print("Average Global Test Loss: ", round(test_loss, 5))
        print("Average Global Training Accuracy: ", round(train_acc, 5))
        print("Average Global Training Loss: ", round(train_loss, 5))
        with open(os.path.join(self.save_path, f"{self.file_name}.csv"), mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([glob_iter, train_loss, test_loss, train_acc, glob_acc])
    
    def plot_graph(self, data, label, output_dir, rounds=None):
        if rounds is None:
            rounds = np.arange(len(data))
        plt.figure()
        plt.plot(rounds, data)
        plt.xlabel("Communication round")
        plt.ylabel(label)
        plt.title(f"{label} vs Communication Rounds")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{label.lower().replace(' ', '_')}_{self.file_name}.png"))
        plt.close()

    
    def plot_results(self):
        rounds = []
        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []
        with open(os.path.join(self.save_path, f"{self.file_name}.csv"), mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                rounds.append(int(row[0]))
                train_loss.append(float(row[1]))
                test_loss.append(float(row[2]))
                train_acc.append(float(row[3]))
                test_acc.append(float(row[4]))
        self.plot_graph(
            train_loss,
            label='Train Loss',
            output_dir=self.save_path,
            rounds=rounds,
        )
        self.plot_graph(
            test_loss,
            label='Test Loss',
            output_dir=self.save_path,
            rounds=rounds,
        )
        self.plot_graph(
            train_acc,
            label='Train Accuracy',
            output_dir=self.save_path,
            rounds=rounds,
        )
        self.plot_graph(
            test_acc,
            label='Test Accuracy',
            output_dir=self.save_path,
            rounds=rounds,
        )
