from flearn.users.user_base import User
from flearn.optimizers.fedoptimizer import *
import random
import torch
from torch.utils.data import RandomSampler, DataLoader
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader, switch_generator
import copy


class UserAVG(User):
    def __init__(self, id, model, train_loader, test_loader, loss_fn_name, local_learning_rate, weight_decay, use_cuda, local_updates, sample_rate, dp, noise_multiplier, max_grad_norm, x_label, y_label, resume=False, checkpoint=None):
        self.model = copy.deepcopy(model)
        optimizer = FedAvgOptimizer(self.model.parameters(), lr=local_learning_rate, weight_decay=weight_decay)
        super().__init__(model, train_loader=train_loader, test_loader=test_loader, loss_fn_name=loss_fn_name, use_cuda=use_cuda, local_updates=local_updates, dp=dp, optimizer=optimizer, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm, id=id, sample_rate=sample_rate, x_label=x_label, y_label=y_label, resume=resume, checkpoint=checkpoint)
        if not dp:
            self.optimizer = optimizer
        self.id = id

    def train_no_dp(self, global_iter):
        if self.use_cuda:
            self.model = self.model.cuda()
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
            batch = next(it)
            X, y = batch[self.x_label], batch[self.y_label]
            if self.use_cuda:
                X, y = X.cuda(), y.cuda()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

        self.model.cpu()
        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()
        
    
    def train_dp(self, global_iter):
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
            loss = self.dp_loss(output, y)
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.model.cpu()
        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()
        
        