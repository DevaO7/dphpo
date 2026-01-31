import torch
from torch.optim import Adam
from torch.optim import Optimizer
import torch.nn as nn
import numpy as np


class FedLOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(FedLOptimizer, self).__init__(params, defaults)


class FedAvgOptimizer(FedLOptimizer):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group.get("weight_decay", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Optional: decoupled weight decay (post-processing; doesn’t use data)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(p.grad, alpha=-lr)
        return loss


class SCAFFOLDOptimizer(FedLOptimizer):
    def __init__(self, params, lr, weight_decay):
        super().__init__(params, lr, weight_decay)

    def step(self, server_controls, user_controls, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group, c, ci in zip(self.param_groups, server_controls, user_controls):
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data + c.data - ci.data
                p.data = p.data - d_p * group['lr']
        return loss
