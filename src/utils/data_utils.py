import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from data.synthetic.data_generator import SyntheticDataset
import os
from typing import Callable, Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
import flwr_datasets.partitioner as partitioners
from flwr_datasets import FederatedDataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from data.synthetic.data_generator import generate_synthetic as generate_synthetic_dataset


@dataclass
class FlowerFederatedLoaders:
    dataset_name: str
    num_clients: int
    partitioner_name: str = "DirichletPartitioner"
    alpha: float = 0.1
    batch_size: int = 64
    num_workers: int = 0
    client_test_fraction: float = 0.2
    seed: int = 42
    image_key: str = "img"
    label_key: str = "label"
    transform_fn: Optional[Callable] = None, 
    max_number_samples_per_client: Optional[int] = 100

    def __post_init__(self) -> None:
        partitioner = getattr(partitioners, self.partitioner_name)
        if self.partitioner_name == "IidPartitioner":
            self.partitioner = partitioner(num_partitions=self.num_clients)
        elif self.partitioner_name == "DirichletPartitioner":
            self.partitioner = partitioner(num_partitions=self.num_clients, partition_by=self.label_key, alpha=self.alpha, seed=self.seed)
        else:
            raise NotImplementedError(f"Partitioner {self.partitioner_name} not implemented.")
        self.fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners={"train": self.partitioner},
        )
        if self.transform_fn is None:
            to_tensor = ToTensor()
            def default_transform(batch):
                batch[self.image_key] = [to_tensor(img) for img in batch[self.image_key]]
                return batch
            self.transform_fn = default_transform

    def get_client_loaders(self, cid: int) -> Tuple[DataLoader, DataLoader]:
        partition = self.fds.load_partition(cid , "train")
        k = min(self.max_number_samples_per_client, len(partition))
        partition = partition.shuffle(seed=self.seed + cid).select(range(k))
        split = partition.train_test_split(
            test_size=self.client_test_fraction,
            seed=self.seed + cid,
        )
        train_ds = split["train"].with_transform(self.transform_fn)
        test_ds = split["test"].with_transform(self.transform_fn)
        g = torch.Generator().manual_seed(self.seed + cid )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=g,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            generator=g
        )

        return train_loader, test_loader
    
    def feature_info(self) -> Dict:
        """Useful when switching datasets: inspect keys/types."""
        part0 = self.fds.load_partition(0, "train")
        return dict(part0.features)  # docs show inspecting partition.features :contentReference[oaicite:6]{index=6}


def _extract_labels_from_batch(
    batch,
    label_key_candidates: Tuple[str, ...] = ("label", "labels", "target", "targets", "y", "character"),
):
    """
    Supports:
      - tuple/list batch: assume labels are at index 1 (x, y) or last position
      - dict batch: find first matching label key
    Returns: torch.Tensor of labels
    """
    if isinstance(batch, (tuple, list)):
        # Common cases: (x, y) or (x, y, meta...)
        y = batch[1] if len(batch) >= 2 else batch[-1]
        return y

    if isinstance(batch, dict):
        for k in label_key_candidates:
            if k in batch:
                return batch[k]
        raise KeyError(f"Could not find labels in batch dict. Keys={list(batch.keys())}")

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def count_label_distribution_from_loaders(
    loaders: Union[Dict[int, DataLoader], List[DataLoader]],
    num_classes: int,
    label_key_candidates: Tuple[str, ...] = ("label", "labels", "target", "targets", "y", "character"),
    max_batches: Optional[int] = None,
) -> Tuple[List[int], np.ndarray]:
    """
    Returns:
      client_ids: list of client IDs in plot order
      counts: shape (num_clients, num_classes)
    """
    if isinstance(loaders, dict):
        client_ids = sorted(loaders.keys())
        loader_list = [loaders[cid] for cid in client_ids]
    else:
        client_ids = list(range(len(loaders)))
        loader_list = loaders

    counts = np.zeros((len(loader_list), num_classes), dtype=np.int64)

    for i, loader in enumerate(loader_list):
        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            y = _extract_labels_from_batch(batch, label_key_candidates=label_key_candidates)

            # Move to CPU numpy-friendly
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu()
            else:
                # e.g. numpy array
                y = torch.as_tensor(y)

            # If one-hot/probabilities: shape (B, C)
            if y.ndim == 2 and y.shape[1] == num_classes:
                y = torch.argmax(y, dim=1)

            # Ensure 1D integer labels
            y = y.view(-1).to(torch.long)

            # Count
            binc = torch.bincount(y, minlength=num_classes).numpy()
            counts[i] += binc

    return client_ids, counts


def plot_stacked_label_distribution(
    client_ids: List[int],
    counts: np.ndarray,
    label_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Per Partition Labels Distribution",
    xlabel: str = "Partition ID",
    ylabel: str = "Count",
    figsize: Tuple[int, int] = (12, 6),
    save_path: str = None
):
    """
    Plots stacked bars per client.
    If normalize=True, plots fractions (each bar sums to 1).
    """
    num_clients, num_classes = counts.shape

    if label_names is None:
        label_names = [f"class_{k}" for k in range(num_classes)]
    assert len(label_names) == num_classes, "label_names must have length num_classes"

    plot_vals = counts.astype(np.float64)
    if normalize:
        denom = plot_vals.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        plot_vals = plot_vals / denom
        ylabel = "Fraction"

    x = np.arange(num_clients)
    bottom = np.zeros(num_clients, dtype=np.float64)

    plt.figure(figsize=figsize)
    for k in range(num_classes):
        plt.bar(x, plot_vals[:, k], bottom=bottom, label=label_names[k])
        bottom += plot_vals[:, k]

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x, [str(cid) for cid in client_ids])
    plt.legend(title="Labels", bbox_to_anchor=(1.02, 0.5), loc="center left")
    plt.savefig(os.path.join(save_path, title.replace(" ", "_") + ".png"))
    plt.close()

def visualize_partition(cfg, train_data_loader, test_data_loader, save_path):
    train_loaders = {}
    test_loaders = {}

    for cid in range(cfg.dataset.nb_users):
        train_loaders[cid] = train_data_loader[cid]
        test_loaders[cid] = test_data_loader[cid]

    # TRAIN distribution
    cids, train_counts = count_label_distribution_from_loaders(
        train_loaders, num_classes=cfg.dataset.dim_output, max_batches=None
    )
    plot_stacked_label_distribution(
        cids, train_counts,
        label_names=None,  # or your class names
        normalize=False,
        title="Per Partition Labels Distribution (Train)", 
        save_path=save_path
    )

    # TEST distribution (client-local)
    cids, test_counts = count_label_distribution_from_loaders(
        test_loaders, num_classes=cfg.dataset.dim_output, max_batches=None
    )
    plot_stacked_label_distribution(
        cids, test_counts,
        label_names=None,
        normalize=False,
        title="Per Partition Labels Distribution (Test)", 
        save_path=save_path
    )

def get_loader_flwr(cfg):
    """
    Loads the raw global datasets.
    """
    fed = FlowerFederatedLoaders(
    dataset_name=cfg.dataset.name,
    num_clients=cfg.dataset.nb_users,
    partitioner_name=cfg.dataset.partitioner_name,
    alpha=cfg.dataset.partitioner_parameter,
    batch_size=cfg.dataset.batch_size,
    num_workers=0,
    client_test_fraction=0.2,
    seed=cfg.run_settings.seed,
    image_key=cfg.dataset.x_label,
    label_key=cfg.dataset.y_label,
    transform_fn=None, 
    max_number_samples_per_client=cfg.dataset.max_samples_per_client
    )

    client_train_loaders = []
    client_test_loaders = []
    print('Flower federated data loaders being created...')
    for user_id in range(cfg.dataset.nb_users):
        train_loader, test_loader = fed.get_client_loaders(user_id)
        client_train_loaders.append(train_loader)
        client_test_loaders.append(test_loader)

    return client_train_loaders, client_test_loaders

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
    X_split, y_split = generate_synthetic_dataset(alpha=cfg.dataset.alpha, 
                                   beta=cfg.dataset.beta,
                                   iid=cfg.dataset.iid,
                                   num_user=cfg.dataset.nb_users,
                                   num_class=cfg.dataset.dim_output,
                                   input_dim=cfg.dataset.dim_input,
                                   num_samples_per_user=cfg.dataset.num_samples,
                                   cluster_seed=cfg.dataset.cluster_seed,
                                   data_seed=cfg.dataset.data_seed
                            )
    client_test_loaders = []
    client_train_loaders = []
    train_size = int(cfg.dataset.num_samples * 0.8)
    for user_id in range(cfg.dataset.nb_users):
        user_data = (torch.as_tensor(X_split[user_id]), torch.as_tensor(y_split[user_id], dtype=torch.long))
        dataset = SyntheticDataset(dataset=user_data)
        g = torch.Generator().manual_seed(cfg.run_settings.seed + user_id)
        train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size], generator=g)
        train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, generator=g, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, generator=g, drop_last=False)
        client_train_loaders.append(train_loader)
        client_test_loaders.append(test_loader)
    return client_train_loaders, client_test_loaders

def get_data_loaders(cfg, train=True, per_client_loader=True):
    """
    Splits training data among clients, returns Global Test Set for server.
    """
    if cfg.dataset.name in ["mnist", "cifar10", "flwrlabs/femnist", "ylecun/mnist", "uoft-cs/cifar10"]:
        train_data_loader, test_data_loader = get_loader_flwr(cfg)
    elif cfg.dataset.name == "synthetic":
        train_data_loader, test_data_loader = get_loader_from_raw_data(cfg, per_client_loader)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name} not supported.")
    return train_data_loader, test_data_loader
