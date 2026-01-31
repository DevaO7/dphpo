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
from data.synthetic.data_generator import generate_data as generate_synthetic_dataset
from simulate import find_optimum, tune_hyperparameters, compile_tuning_results
from utils.data_utils import get_data_loaders

def generate_data(cfg):
    if cfg.dataset.name == 'synthetic':
        generate_synthetic_dataset(num_users=cfg.dataset.nb_users, 
                            same_sample_size=cfg.dataset.same_sample_size, 
                            num_samples=cfg.dataset.num_samples, 
                            dim_input=cfg.dataset.dim_input, 
                            dim_output=cfg.dataset.dim_output, 
                            noise_ratio=cfg.dataset.noise_ratio, 
                            alpha=cfg.dataset.alpha, 
                            beta=cfg.dataset.beta, 
                            ratio_training=cfg.dataset.ratio_training, 
                            number=cfg.dataset.number,
                            iid=cfg.dataset.iid
                            )
    else: 
        raise NotImplementedError(f"Dataset {cfg.dataset.name} is not supported.")


def simulate(cfg):
    pass

def compile_results(cfg):
    pass

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_simulation(cfg: DictConfig):

    if cfg.run_mode.generate_data:
        generate_data(cfg)

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