from flearn.users.user_base import User
from flearn.optimizers.fedoptimizer import *
import random
import torch
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import SubsetRandomSampler
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader, switch_generator
import copy


class UserAVG(User):
    def __init__(self, id, model, train_loader, test_loader, loss_fn_name, local_learning_rate, weight_decay, use_cuda, local_updates, sample_rate, dp, noise_multiplier, max_grad_norm, x_label, y_label, resume=False, checkpoint=None, data_sampling_scheme='fixed_size_sampling'):
        self.model = copy.deepcopy(model)
        optimizer = FedAvgOptimizer(self.model.parameters(), lr=local_learning_rate, weight_decay=weight_decay)
        super().__init__(model, train_loader=train_loader, test_loader=test_loader, loss_fn_name=loss_fn_name, use_cuda=use_cuda, local_updates=local_updates, dp=dp, optimizer=optimizer, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm, id=id, sample_rate=sample_rate, x_label=x_label, y_label=y_label, resume=resume, checkpoint=checkpoint, data_sampling_scheme=data_sampling_scheme)
        if not dp:
            self.optimizer = optimizer
        self.id = id

    def train_poisson_sampling(self, global_iter, dp):
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model.train()
        g = torch.Generator().manual_seed(self.id + global_iter * 100)
        self.dp_train_loader = switch_generator(data_loader=self.dp_train_loader, generator=g)

        it = iter(self.dp_train_loader)
        for step in range(1, self.local_updates + 1):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.dp_train_loader)
                batch = next(it)
            X, y = batch[self.x_label], batch[self.y_label] 
            if y.numel() == 0:
                continue
            if self.use_cuda:
                X, y = X.cuda(), y.cuda()
            self.optimizer.zero_grad()
            output = self.model(X)
            if dp:
                loss = self.dp_loss(output, y)
            else:
                loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.model.cpu()
        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()

    def train_fixed_size_sampling(self, global_iter, dp):
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model.train()
        train_idx = np.arange(self.train_samples)
        for step in range(1, self.local_updates + 1):
            np.random.seed(self.id + global_iter * 100 + step)
            torch.manual_seed(self.id + global_iter * 100 + step)
            train_sampler = SubsetRandomSampler(train_idx)
            it = iter(DataLoader(self.traindataset, self.batch_size, sampler=train_sampler))
            batch = next(it)
            X, y = batch[self.x_label], batch[self.y_label]
            if y.numel() == 0:
                continue
            if self.use_cuda:
                X, y = X.cuda(), y.cuda()
            self.optimizer.zero_grad()
            output = self.model(X)
            if dp:
                loss = self.dp_loss(output, y)
            else:
                loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.model.cpu()
        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()
