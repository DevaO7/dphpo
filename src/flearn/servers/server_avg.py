from flearn.servers.server_base import Server
from flearn.users.user_avg import UserAVG
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

class FedAvg(Server):
    def __init__(self, model, train_data_loader, test_data_loader, num_glob_iters, save_path, loss_fn_name, local_learning_rate, global_learning_rate, weight_decay, use_cuda, similarity, file_name, client_ratio, dp, local_updates, sample_rate, noise_multiplier, max_grad_norm, x_label, y_label, sampling_scheme):
        super().__init__(model, similarity, save_path, file_name, client_ratio, dp, use_cuda, num_glob_iters, sampling_scheme)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        
        self.global_learning_rate = global_learning_rate
        checkpoint_path = os.path.join('checkpoints', save_path, f"checkpoint.pth")
        resume = os.path.exists(checkpoint_path) 
        self.num_users = len(train_data_loader)
        if resume:
            print(f"Resuming from checkpoint at round {self.start_iter}")
            truncate_csv_file(csv_path=os.path.join(self.save_path, f"{self.file_name}.csv"), keep_round=self.start_iter-1)
        else:
            with open(os.path.join(self.save_path, f"{self.file_name}.csv"), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Round", "Train Loss", "Test Loss", "Train Accuracy", "Test Accuracy"])  # Column Headers

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
                resume=resume, 
                checkpoint=self.checkpoint if resume else None, 
                sampling_scheme=sampling_scheme
            )
            self.users.append(user)

    def train(self):
        for glob_iter in range(self.start_iter, self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            self.send_parameters()
            self.evaluate(glob_iter)
            if self.sampling_scheme == 'fixed_size':
                 self.selected_users = self.select_users_fixed_sampling(glob_iter)
            elif self.sampling_scheme == 'poisson_sampling':
                self.selected_users = self.select_users_poisson_sampling(glob_iter)
            if len(self.selected_users) == 0:
                print("No users selected, skipping this round.")
                continue
            for user in self.selected_users:
                if self.dp: 
                    user.train_dp(glob_iter)
                else:
                    user.train_no_dp(glob_iter)
            self.aggregate_parameters()
            if glob_iter % 10 == 0:
                self.save_checkpoint(glob_iter)
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