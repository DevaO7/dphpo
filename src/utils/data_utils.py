import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
from byzfl import Client, Server, ByzantineClient, DataDistributor
import byzfl.aggregators as aggregators
from data.synthetic.data_generator import SyntheticDataset, read_data
import byzfl.fed_framework.models as models
import os


def read_user_data(index, raw_data, dataset):
    """Returns:
        id: id of user
        train_data: list of (data, labels) for training
        test_data: list of (data, labels) for testing
    """
    id = raw_data[0][index]
    data = raw_data[1][id]
    X, y = data['x'],  data['y']
    if dataset == "CIFAR-10":
        X = torch.as_tensor(X).view(-1, 3, 32, 32).type(torch.float32)
        y = torch.as_tensor(y).type(torch.int64)
    else:
        # image flattened for FEMNIST, MNIST
        X = torch.as_tensor(X).type(torch.float32)
        y = torch.as_tensor(y).type(torch.int64)
    return id, (X, y)

def get_loader_byzfl(cfg):
    """
    Loads the raw global datasets.
    """
    data_path = cfg.dataset.get("path", "./data")
    
    if cfg.dataset.name == "mnist":
        stats = ((0.1307,), (0.3081,))
    elif cfg.dataset.name == "cifar10":
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name} not supported.")
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
    ])
    train_dataset = getattr(datasets, cfg.dataset.name.upper())(data_path, train=True, download=True, transform=transform)
    test_dataset = getattr(datasets, cfg.dataset.name.upper())(data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    if cfg.run_settings.attack_need_data:
        nb_recipients = cfg.dataset.nb_users
    else:
        nb_recipients = cfg.dataset.nb_users - cfg.run_settings.nb_byzantine

    dist_params = {
        "data_distribution_name": cfg.dataset.distribution_name,
        "distribution_parameter": cfg.dataset.distribution_parameter,
        "nb_honest": nb_recipients, 
        "batch_size": cfg.dataset.batch_size,
        "data_loader": train_loader
    }

    print(f"Initializing Data Distributor (Mode: {cfg.dataset.distribution_name})...")
    distributor = DataDistributor(dist_params)
    
    client_train_loaders = distributor.split_data()

    return client_train_loaders, test_loader

def get_per_client_loader(cfg, data):
    # For FL Experiments
    data_loader = []
    print('Per-client data loaders being created...')
    for user_id in range(cfg.dataset.nb_users):
        _, user_train_data = read_user_data(user_id, data, cfg.dataset.name)
        dataset = SyntheticDataset(dataset=user_train_data)
        g = torch.Generator().manual_seed(cfg.run_settings.seed + user_id)
        # This is to account for DP sampling
        train_loader = DataLoader(dataset, batch_size=int(cfg.server.sampling_rate*len(dataset)) if cfg.server.dp else cfg.run_settings.batch_size, shuffle=True, generator=g, drop_last=False)
        data_loader.append(train_loader)
    return data_loader

def get_global_loader(cfg, data):
    # For Find_Optimum
    data_X = []
    data_y = []
    for user_id in range(cfg.dataset.nb_users):
        _, user_data = read_user_data(user_id, data, cfg.dataset.name)
        data_X.append(user_data[0])
        data_y.append(user_data[1])
    data = (torch.cat(data_X, dim=0), torch.cat(data_y, dim=0))
    global_dataset = SyntheticDataset(
        dataset=data
    )
    global_loader = DataLoader(global_dataset, batch_size=cfg.run_settings.batch_size, shuffle=True)
    return global_loader

def get_loader_from_raw_data(cfg, per_client_loader=True):
    if cfg.dataset.iid:
        similarity = "iid"
    else:
        similarity = str((cfg.dataset.alpha, cfg.dataset.beta))
    user_ids, _, train_data, test_data = read_data(number=str(cfg.dataset.number), similarity=similarity, dim_pca=None)
    if per_client_loader: 
        train_data_loader = get_per_client_loader(cfg, (user_ids, train_data))
        test_data_loader = get_per_client_loader(cfg, (user_ids, test_data))
    else: 
        train_data_loader = get_global_loader(cfg, (user_ids, train_data))
        test_data_loader = get_global_loader(cfg, (user_ids, test_data))
    return train_data_loader, test_data_loader
        

def get_data_loaders(cfg, train=True, per_client_loader=True):
    """
    Splits training data among clients, returns Global Test Set for server.
    """
    if cfg.dataset.name in ["mnist", "cifar10"]:
        train_data_loader, test_data_loader = get_loader_byzfl(cfg)
    elif cfg.dataset.name == "synthetic":
        train_data_loader, test_data_loader = get_loader_from_raw_data(cfg, per_client_loader)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name} not supported.")
    return train_data_loader, test_data_loader
