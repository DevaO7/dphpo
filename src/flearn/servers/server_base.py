import numpy as np
import os 
import csv
import copy
import matplotlib.pyplot as plt
import torch


class Server:
    def __init__(self, model, similarity, save_path, file_name, client_ratio, dp, use_cuda, num_glob_iters, resume=False):
        self.users = []
        self.selected_users = []
        self.use_cuda = use_cuda
        if resume:
            checkpoint_path = os.path.join('checkpoints', save_path, f"checkpoint.pth")
            if os.path.exists(checkpoint_path):
                self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.model = copy.deepcopy(model)
                self.model.load_state_dict(self.checkpoint['model_state_dict'])
                self.start_iter = self.checkpoint['round'] + 1
            else:
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        else:
            self.model = copy.deepcopy(model)
            self.start_iter = 0
        self.similarity = similarity
        self.save_path = save_path
        self.file_name = file_name
        self.client_ratio = client_ratio
        self.dp = dp
        self.num_glob_iters = num_glob_iters

    def send_parameters(self):
        """Users setting their parameters from the server."""
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)
    
    def save_checkpoint(self, glob_iter):
        checkpoint_path = os.path.join('checkpoints', self.save_path, f"checkpoint.pth")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        privacy_engine_generator = {}
        optimizer_state_dict = {}
        if self.dp:
            for user in self.users:
                privacy_engine_generator[user.id] = user.generator.get_state()
                optimizer_state_dict[user.id] = user.optimizer.state_dict()
        check_point = {
            'round': glob_iter,
            'model_state_dict': self.model.state_dict(),
            'privacy_engine_generator': privacy_engine_generator,
            'optimizer_state_dict': optimizer_state_dict
        }
        torch.save(check_point, checkpoint_path)

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

    def select_users(self, glob_iter):
        assert 0.0 < self.client_ratio <= 1.0
        ids = [c.id for c in self.users]
        probs = np.ones(len(self.users))*self.client_ratio
        selected_ids = self.poisson_sampling(ids, probs, seed=glob_iter)
        print(f"Selected users: {selected_ids}")
        selected_set = set(map(int, selected_ids.tolist()))
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
    
    def plot_graph(self, data, label='Train Loss', output_dir=None):
        rounds = np.arange(self.num_glob_iters)
        plt.figure()
        plt.plot(rounds, data)
        plt.xlabel("Communication round")
        plt.ylabel(label)
        plt.title(f"{label} vs Communication Rounds")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{label.lower().replace(' ', '_')}_{self.file_name}.png"))
        plt.close()

    
    def plot_results(self):
        train_loss = np.zeros(self.num_glob_iters)
        test_loss = np.zeros(self.num_glob_iters)
        train_acc = np.zeros(self.num_glob_iters)
        test_acc = np.zeros(self.num_glob_iters)
        with open(os.path.join(self.save_path, f"{self.file_name}.csv"), mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                round_num = int(row[0])
                train_loss[round_num] = float(row[1])
                test_loss[round_num] = float(row[2])
                train_acc[round_num] = float(row[3])
                test_acc[round_num] = float(row[4])
        self.plot_graph(train_loss, label='Train Loss', output_dir=self.save_path)
        self.plot_graph(test_loss, label='Test Loss', output_dir=self.save_path)
        self.plot_graph(train_acc, label='Train Accuracy', output_dir=self.save_path)
        self.plot_graph(test_acc, label='Test Accuracy', output_dir=self.save_path)
        
        