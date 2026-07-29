from flearn.servers.server_base import Server
from flearn.users.user_avg import UserAVG
from utils.seed_utils import DP_NOISE_STREAM, derive_seed
import csv
import os

def truncate_csv_file(csv_path: str, keep_round: int) -> None:
    if not os.path.exists(csv_path):
        return
    tmp_path = csv_path + ".tmp"
    with open(csv_path, "r", newline="") as src, open(tmp_path, "w", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader, None)
        if header is not None:
            writer.writerow(header)
        for row in reader:
            if not row:
                continue
            try:
                r = int(row[0])
            except ValueError:
                writer.writerow(row)
                continue
            if r <= keep_round:
                writer.writerow(row)
            else:
                break
    os.replace(tmp_path, csv_path)


def validate_csv_rounds(
    csv_path: str,
    completed_rounds: int,
    start_round: int = 0,
) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Metrics CSV does not exist: {csv_path}"
        )

    observed_rounds = []
    with open(csv_path, "r", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if not header or header[0] != "Round":
            raise RuntimeError(
                f"Metrics CSV {csv_path} is missing its Round header."
            )

        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                observed_rounds.append(int(row[0]))
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Metrics CSV {csv_path} has an invalid round at "
                    f"line {row_number}: {row[0]!r}."
                ) from error

    expected_rounds = list(range(start_round, completed_rounds))
    if observed_rounds[:len(expected_rounds)] != expected_rounds:
        raise RuntimeError(
            f"Metrics CSV {csv_path} does not contain contiguous data "
            f"for rounds {start_round} through {completed_rounds - 1}. "
            f"Observed rounds: {observed_rounds!r}."
        )


class FedAvg(Server):
    def __init__(self, model, train_data_loader, test_data_loader, num_glob_iters, save_path, loss_fn_name, local_learning_rate, global_learning_rate, weight_decay, use_cuda, similarity, file_name, client_ratio, dp, local_updates, sample_rate, noise_multiplier, max_grad_norm, x_label, y_label, client_sampling_scheme, data_sampling_scheme, stage=None, stage_1_end=None, base_seed=0):
        super().__init__(model, similarity, save_path, file_name, client_ratio, dp, use_cuda, num_glob_iters, client_sampling_scheme, data_sampling_scheme, stage, stage_1_end, base_seed)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.global_learning_rate = global_learning_rate

        csv_path = os.path.join(
            self.save_path,
            f"{self.file_name}.csv",
        )

        if self.start_iter > self.num_glob_iters:
            raise RuntimeError(
                f"Checkpoint records {self.start_iter} completed rounds, "
                f"but num_glob_iters is only {self.num_glob_iters}."
            )
        if (
            self.stage == 1
            and self.stage_1_end is not None
            and self.num_glob_iters != self.stage_1_end
        ):
            raise ValueError(
                "For Stage 1, num_glob_iters must equal stage_1_end; "
                f"got {self.num_glob_iters} and {self.stage_1_end}."
            )
        if self.stage == 2 and self.num_glob_iters < self.stage_1_end:
            raise ValueError(
                "For Stage 2, num_glob_iters must be at least "
                f"stage_1_end ({self.stage_1_end}); got "
                f"{self.num_glob_iters}."
            )

        if self.stage == 2:
            stage_1_csv_path = os.path.join(
                self.save_path,
                "stage_1.csv",
            )
            validate_csv_rounds(
                csv_path=stage_1_csv_path,
                completed_rounds=self.stage_1_end,
            )

        if self.resume_from_checkpoint:
            validate_csv_rounds(
                csv_path=csv_path,
                completed_rounds=self.start_iter,
                start_round=(
                    self.stage_1_end
                    if self.stage == 2
                    else 0
                ),
            )
            print(
                f"Resuming Stage {self.stage} from checkpoint after "
                f"{self.start_iter} completed rounds"
            )
            truncate_csv_file(
                csv_path=csv_path,
                keep_round=self.start_iter - 1,
            )
        elif self.initialized_from_stage_1:
            print(
                "Initializing Stage 2 model from the completed Stage 1 "
                f"checkpoint after {self.stage_1_end} rounds"
            )
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Round", "Train Loss", "Test Loss", "Train Accuracy", "Test Accuracy"])
        else:
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Round", "Train Loss", "Test Loss", "Train Accuracy", "Test Accuracy"])  # Column Headers

        self.num_users = len(train_data_loader)

        # Initialize users
        for id in range(self.num_users):
            user = UserAVG(
                id=id,
                model=model,
                train_loader=train_data_loader[id],
                test_loader=test_data_loader[id],
                loss_fn_name=loss_fn_name, 
                local_learning_rate=local_learning_rate,
                weight_decay=weight_decay, 
                use_cuda=use_cuda, 
                local_updates=local_updates, 
                sample_rate=sample_rate, 
                dp=dp, 
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm, 
                x_label=x_label,
                y_label=y_label, 
                resume=self.resume_from_checkpoint,
                checkpoint=(
                    self.checkpoint
                    if self.resume_from_checkpoint
                    else None
                ),
                data_sampling_scheme=data_sampling_scheme,
                base_seed=self.base_seed,
                dp_noise_seed=(
                    derive_seed(
                        self.base_seed,
                        self.stage,
                        id,
                        stream=DP_NOISE_STREAM,
                    )
                    if self.stage is not None
                    else None
                ),
            )
            self.users.append(user)

    def train(self):
        for glob_iter in range(self.start_iter, self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            self.send_parameters()
            self.evaluate(glob_iter)
            if self.client_sampling_scheme == 'fixed_size_sampling':
                 self.selected_users = self.select_users_fixed_sampling(glob_iter)
            elif self.client_sampling_scheme == 'poisson_sampling':
                self.selected_users = self.select_users_poisson_sampling(glob_iter)
            if len(self.selected_users) == 0:
                print("No users selected, skipping this round.")
                if (glob_iter + 1) % 10 == 0:
                    self.save_checkpoint(glob_iter + 1)
                continue
            for user in self.selected_users:
                if self.data_sampling_scheme == 'poisson_sampling':
                    user.train_poisson_sampling(glob_iter, self.dp)
                else:
                    user.train_fixed_size_sampling(glob_iter, self.dp)
            self.aggregate_parameters()
            if (glob_iter + 1) % 10 == 0:
                self.save_checkpoint(glob_iter + 1)

        # Always persist the exact stage boundary, even when it is not a
        # multiple of the periodic checkpoint interval.
        self.save_checkpoint(self.num_glob_iters)
        self.plot_results()
    
    def aggregate_parameters(self):
        """Aggregation update of the server model."""
        assert (self.users is not None and len(self.users) > 0)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
    
    def add_parameters(self, user, ratio):
        """Adding to the server model the contribution term from user."""
        for server_param, del_model in zip(self.model.parameters(), user.delta_model):
            server_param.data = server_param.data + self.global_learning_rate * del_model.data * ratio
