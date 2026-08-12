"""Centralized dataset loading for the HPO experiments."""

from collections import Counter
from functools import lru_cache
from pathlib import Path
import re
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import datasets as vision_datasets
from torchvision import transforms

_DATASET_ALIASES = {
    "mnist": "mnist",
    "fashion_mnist": "fashion_mnist",
    "fashionmnist": "fashion_mnist",
    "cifar_10": "cifar10",
    "cifar10": "cifar10",
    "imdb": "imdb",
    "synthetic": "synthetic",
}
_VISION_DATASETS = {
    "mnist": vision_datasets.MNIST,
    "fashion_mnist": vision_datasets.FashionMNIST,
    "cifar10": vision_datasets.CIFAR10,
}
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
_PAD_TOKEN_ID = 0
_UNKNOWN_TOKEN_ID = 1


class _HuggingFaceSequenceDataset(Dataset):
    """Expose a tokenized Hugging Face split as PyTorch tuples."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        return (
            torch.as_tensor(example["input_ids"], dtype=torch.long),
            torch.as_tensor(example["label"], dtype=torch.long),
        )


def _get_dataset_config(config: DictConfig):
    if "experiment" not in config or "dataset" not in config.experiment:
        raise ValueError(
            "The centralized experiment configuration must define "
            "experiment.dataset."
        )
    dataset_config = config.experiment.dataset
    if "name" not in dataset_config:
        raise ValueError("experiment.dataset.name must be defined.")
    return dataset_config


def _normalize_dataset_name(name: str) -> str:
    normalized_name = str(name).strip().lower().replace("-", "_")
    try:
        return _DATASET_ALIASES[normalized_name]
    except KeyError as error:
        supported = ", ".join(sorted(set(_DATASET_ALIASES.values())))
        raise ValueError(
            f"Unsupported centralized dataset {name!r}. "
            f"Supported datasets: {supported}."
        ) from error


def _get_positive_integer(config, key, default):
    value = config.get(key, default)
    if isinstance(value, bool):
        raise ValueError(
            f"experiment.dataset.{key} must be a positive integer; "
            f"got {value!r}."
        )
    try:
        parsed_value = int(value)
        value_as_float = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"experiment.dataset.{key} must be a positive integer; "
            f"got {value!r}."
        ) from error
    if parsed_value < 1 or value_as_float != parsed_value:
        raise ValueError(
            f"experiment.dataset.{key} must be a positive integer; "
            f"got {value!r}."
        )
    return parsed_value


def _resolve_loader_seed(config: DictConfig, seed: Optional[int]) -> int:
    if seed is None:
        seed = config.experiment.get(
            "seed",
            config.get("run_settings", {}).get("seed", 0),
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(
            "The centralized data-loader seed must be a non-negative "
            f"integer; got {seed!r}."
        )
    return seed


@lru_cache(maxsize=8)
def _load_vision_datasets(name, data_root, download):
    dataset_class = _VISION_DATASETS[name]
    tensor_transform = transforms.ToTensor()
    train_dataset = dataset_class(
        root=data_root,
        train=True,
        transform=tensor_transform,
        download=download,
    )
    test_dataset = dataset_class(
        root=data_root,
        train=False,
        transform=tensor_transform,
        download=download,
    )
    return train_dataset, test_dataset


@lru_cache(maxsize=8)
def _load_synthetic_datasets(
    num_samples,
    input_dim,
    num_classes,
    alpha,
    beta,
    iid,
    cluster_seed,
    data_seed,
    train_fraction,
    split_seed,
):
    from data.synthetic.data_generator import generate_synthetic

    features_by_user, labels_by_user = generate_synthetic(
        alpha=alpha,
        beta=beta,
        iid=iid,
        num_user=1,
        num_class=num_classes,
        input_dim=input_dim,
        num_samples_per_user=num_samples,
        cluster_seed=cluster_seed,
        data_seed=data_seed,
    )
    dataset = TensorDataset(
        torch.as_tensor(features_by_user[0], dtype=torch.float32),
        torch.as_tensor(labels_by_user[0], dtype=torch.long),
    )
    train_size = int(len(dataset) * train_fraction)
    if train_size < 1 or train_size >= len(dataset):
        raise ValueError(
            "experiment.dataset.train_fraction must produce non-empty "
            "train and test splits."
        )
    split_generator = torch.Generator().manual_seed(split_seed)
    return random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=split_generator,
    )


def _tokenize(text: str):
    return _TOKEN_PATTERN.findall(text.lower())


def _build_vocabulary(texts, vocabulary_size, minimum_frequency):
    token_counts = Counter()
    for text in texts:
        token_counts.update(_tokenize(text))

    vocabulary = {}
    for token, count in token_counts.most_common():
        if count < minimum_frequency:
            break
        if len(vocabulary) >= vocabulary_size - 2:
            break
        vocabulary[token] = len(vocabulary) + 2
    return vocabulary


def _encode_text(text, vocabulary, maximum_length):
    token_ids = [
        vocabulary.get(token, _UNKNOWN_TOKEN_ID)
        for token in _tokenize(text)[:maximum_length]
    ]
    token_ids.extend(
        [_PAD_TOKEN_ID] * (maximum_length - len(token_ids))
    )
    return token_ids


@lru_cache(maxsize=4)
def _load_imdb_datasets(
    dataset_id,
    vocabulary_size,
    minimum_frequency,
    maximum_length,
    cache_directory,
):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "Loading IMDB requires the Hugging Face 'datasets' package."
        ) from error

    dataset = load_dataset(dataset_id, cache_dir=cache_directory)
    if "train" not in dataset or "test" not in dataset:
        raise ValueError(
            f"Hugging Face dataset {dataset_id!r} must provide "
            "train and test splits."
        )
    required_columns = {"text", "label"}
    for split_name in ("train", "test"):
        missing_columns = required_columns.difference(
            dataset[split_name].column_names
        )
        if missing_columns:
            raise ValueError(
                f"Hugging Face dataset {dataset_id!r} split "
                f"{split_name!r} is missing columns "
                f"{sorted(missing_columns)}."
            )

    vocabulary = _build_vocabulary(
        dataset["train"]["text"],
        vocabulary_size=vocabulary_size,
        minimum_frequency=minimum_frequency,
    )

    def encode_batch(batch):
        return {
            "input_ids": [
                _encode_text(text, vocabulary, maximum_length)
                for text in batch["text"]
            ]
        }

    tokenized_datasets = {}
    for split_name in ("train", "test"):
        tokenized_datasets[split_name] = dataset[split_name].map(
            encode_batch,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing {dataset_id} {split_name}",
        )

    return (
        _HuggingFaceSequenceDataset(tokenized_datasets["train"]),
        _HuggingFaceSequenceDataset(tokenized_datasets["test"]),
    )


def _load_datasets(dataset_config):
    name = _normalize_dataset_name(dataset_config.name)
    data_root = str(
        Path(dataset_config.get("root", "data/central")).expanduser()
    )

    if name in _VISION_DATASETS:
        return _load_vision_datasets(
            name=name,
            data_root=data_root,
            download=bool(dataset_config.get("download", True)),
        )

    if name == "synthetic":
        train_fraction = float(
            dataset_config.get("train_fraction", 0.8)
        )
        if not 0 < train_fraction < 1:
            raise ValueError(
                "experiment.dataset.train_fraction must be between "
                f"0 and 1; got {train_fraction!r}."
            )
        return _load_synthetic_datasets(
            num_samples=_get_positive_integer(
                dataset_config, "num_samples", 10_000
            ),
            input_dim=_get_positive_integer(
                dataset_config, "input_dim", 60
            ),
            num_classes=_get_positive_integer(
                dataset_config, "num_classes", 10
            ),
            alpha=float(dataset_config.get("alpha", 0.0)),
            beta=float(dataset_config.get("beta", 0.0)),
            iid=bool(dataset_config.get("iid", True)),
            cluster_seed=int(dataset_config.get("cluster_seed", 0)),
            data_seed=int(dataset_config.get("data_seed", 0)),
            train_fraction=train_fraction,
            split_seed=int(dataset_config.get("split_seed", 0)),
        )

    return _load_imdb_datasets(
        dataset_id=str(
            dataset_config.get("dataset_id", "stanfordnlp/imdb")
        ),
        vocabulary_size=_get_positive_integer(
            dataset_config, "vocabulary_size", 20_000
        ),
        minimum_frequency=_get_positive_integer(
            dataset_config, "minimum_token_frequency", 1
        ),
        maximum_length=_get_positive_integer(
            dataset_config, "maximum_sequence_length", 256
        ),
        cache_directory=data_root,
    )


def get_data_loaders(
    config: DictConfig,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Return deterministic central train and test data loaders.

    The dataset is selected exclusively by ``experiment.dataset.name``.
    Supported canonical names are ``synthetic``, ``mnist``,
    ``fashion_mnist``, ``cifar10``, and ``imdb``.
    """
    dataset_config = _get_dataset_config(config)
    loader_seed = _resolve_loader_seed(config, seed)
    batch_size = _get_positive_integer(
        dataset_config, "batch_size", 32
    )
    num_workers = int(dataset_config.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError(
            "experiment.dataset.num_workers must be non-negative; "
            f"got {num_workers!r}."
        )

    train_dataset, test_dataset = _load_datasets(dataset_config)
    common_loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(dataset_config.get("pin_memory", False)),
        "persistent_workers": (
            bool(dataset_config.get("persistent_workers", False))
            if num_workers > 0
            else False
        ),
    }
    train_generator = torch.Generator().manual_seed(loader_seed)
    train_loader = DataLoader(
        train_dataset,
        shuffle=bool(dataset_config.get("shuffle_train", True)),
        drop_last=bool(dataset_config.get("drop_last_train", False)),
        generator=train_generator,
        **common_loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_options,
    )
    return train_loader, test_loader
