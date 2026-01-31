from flearn.users.user_base import User
from flearn.optimizers.fedoptimizer import *
import random
import torch
from torch.utils.data import RandomSampler, DataLoader
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader, switch_generator
import copy


class UserAVG(User):
    def __init__(self, id, model, train_loader, test_loader, loss_fn, local_learning_rate, weight_decay, use_cuda, local_updates, sample_rate, dp, noise_multiplier, max_grad_norm):
        self.model = copy.deepcopy(model)
        optimizer = FedAvgOptimizer(self.model.parameters(), lr=local_learning_rate, weight_decay=weight_decay)
        super().__init__(model, train_loader=train_loader, test_loader=test_loader, loss_fn=loss_fn, use_cuda=use_cuda, local_updates=local_updates, dp=dp, optimizer=optimizer, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm, id=id)
        if not dp:
            self.optimizer = optimizer
        self.id = id
        self.loss = loss_fn

    def train_no_dp(self, global_iter):
        self.model.train()
        sampler_g = torch.Generator().manual_seed(self.id + global_iter*100)
        sampler = RandomSampler(
                    self.traindataset,
                    replacement=True,
                    generator=sampler_g, 
                    num_samples=self.local_updates*self.batch_size
                )
        train_loader = DataLoader(self.traindataset, batch_size=self.batch_size, sampler=sampler, shuffle=False, drop_last=True)
        it = iter(train_loader)
        for step in range(1, self.local_updates + 1):
            X, y = next(it)
            if self.use_cuda:
                X, y = X.cuda(), y.cuda()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()
    
    def train_dp(self, global_iter):
        self.model.train()
        g = torch.Generator().manual_seed(self.id + global_iter * 100)
        self.dp_train_loader = switch_generator(data_loader=self.dp_train_loader, generator=g)

        it = iter(self.dp_train_loader)
        for step in range(1, self.local_updates + 1):
            try:
                X, y = next(it)
            except StopIteration:
                it = iter(self.dp_train_loader)
                X, y = next(it)
            if y.numel() == 0:
                continue
            if self.use_cuda:
                X, y = X.cuda(), y.cuda()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()
        