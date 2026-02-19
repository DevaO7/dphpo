import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from simulate import find_optimum, tune_hyperparameters, compile_tuning_results
from utils.data_utils import get_data_loaders


def simulate(cfg):
    pass

def compile_results(cfg):
    pass

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_simulation(cfg: DictConfig):

    if cfg.run_mode.find_optimum:
        global_train_loader, global_test_loader = get_data_loaders(cfg, per_client_loader=False)
        find_optimum(cfg, train_loader=global_train_loader, test_loader=global_test_loader)

    if cfg.run_mode.tune_hyperparameter:
        tune_hyperparameters(cfg)

    if cfg.run_mode.compile_tuning_results:
        compile_tuning_results(cfg)

    if cfg.run_mode.simulate:
        simulate(cfg)

    if cfg.run_mode.compile_results:
        compile_results(cfg)

if __name__ == "__main__":
    run_simulation()