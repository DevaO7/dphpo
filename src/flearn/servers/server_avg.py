from flearn.servers.server_base import Server
from flearn.users.user_avg import UserAVG
import csv
import os

class FedAvg(Server):
    def __init__(self, model, train_data_loader, test_data_loader, num_glob_iters, save_path, loss_fn, local_learning_rate, global_learning_rate, weight_decay, use_cuda, similarity, file_name, client_ratio, dp, local_updates, sample_rate, noise_multiplier, max_grad_norm):
        super().__init__(model, similarity, save_path, file_name, client_ratio, dp, use_cuda, num_glob_iters)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        
        self.global_learning_rate = global_learning_rate

        self.num_users = len(train_data_loader)
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
                loss_fn=loss_fn, 
                local_learning_rate=local_learning_rate,
                weight_decay=weight_decay, 
                use_cuda=use_cuda, 
                local_updates=local_updates, 
                sample_rate=sample_rate, 
                dp=dp, 
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm
            )
            self.users.append(user)

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            self.send_parameters()
            self.evaluate(glob_iter)
            self.selected_users = self.select_users(glob_iter)
            if len(self.selected_users) == 0:
                print("No users selected, skipping this round.")
                continue
            for user in self.selected_users:
                if self.dp: 
                    user.train_dp(glob_iter)
                else:
                    user.train_no_dp(glob_iter)
            self.aggregate_parameters()
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