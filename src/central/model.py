"""Dataset-specific models for centralized HPO experiments."""

import math

import torch
from torch import nn


_DATASET_ALIASES = {
    "mnist": "mnist",
    "fashion_mnist": "fashion_mnist",
    "fashionmnist": "fashion_mnist",
}


class TemperedSigmoid(nn.Module):
    """Bounded activation used by the paper's image classifiers.

    The activation is ``scale * sigmoid(inverse_temperature * x) - offset``.
    Its paper defaults (2, 2, 1) are exactly equivalent to ``tanh(x)``.
    """

    def __init__(
        self,
        scale=2.0,
        inverse_temperature=2.0,
        offset=1.0,
    ):
        super().__init__()
        self.scale = _positive_finite_float(scale, "scale")
        self.inverse_temperature = _positive_finite_float(
            inverse_temperature,
            "inverse_temperature",
        )
        self.offset = _finite_float(offset, "offset")

    def forward(self, inputs):
        return (
            self.scale
            * torch.sigmoid(self.inverse_temperature * inputs)
            - self.offset
        )

    def extra_repr(self):
        return (
            f"scale={self.scale}, "
            f"inverse_temperature={self.inverse_temperature}, "
            f"offset={self.offset}"
        )


class MNISTCNN(nn.Module):
    """Opacus example CNN with tempered-sigmoid activations.

    For ten output classes, this model has 26,010 trainable parameters,
    matching the approximately 26k parameters reported in the paper.
    """

    def __init__(
        self,
        num_classes=10,
        activation_parameters=None,
    ):
        super().__init__()
        num_classes = _positive_integer(num_classes, "num_classes")
        activation_parameters = activation_parameters or {}

        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.activation1 = TemperedSigmoid(**activation_parameters)
        self.activation2 = TemperedSigmoid(**activation_parameters)
        self.activation3 = TemperedSigmoid(**activation_parameters)
        self.pool = nn.MaxPool2d(2, 1)
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        outputs = self.activation1(self.conv1(inputs))
        outputs = self.pool(outputs)
        outputs = self.activation2(self.conv2(outputs))
        outputs = self.pool(outputs)
        outputs = self.flatten(outputs)
        outputs = self.activation3(self.fc1(outputs))
        return self.fc2(outputs)


class FashionMNISTMLP(nn.Module):
    """Three-layer paper MLP with two width-120 hidden layers.

    For ten output classes, this model has 109,930 trainable parameters,
    matching the approximately 109k parameters reported in the paper.
    """

    def __init__(
        self,
        num_classes=10,
        hidden_width=120,
        activation_parameters=None,
    ):
        super().__init__()
        num_classes = _positive_integer(num_classes, "num_classes")
        hidden_width = _positive_integer(hidden_width, "hidden_width")
        activation_parameters = activation_parameters or {}

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.fc3 = nn.Linear(hidden_width, num_classes)
        self.activation1 = TemperedSigmoid(**activation_parameters)
        self.activation2 = TemperedSigmoid(**activation_parameters)

    def forward(self, inputs):
        outputs = self.flatten(inputs)
        outputs = self.activation1(self.fc1(outputs))
        outputs = self.activation2(self.fc2(outputs))
        return self.fc3(outputs)


def count_parameters(model, trainable_only=False):
    """Return the number of scalar model parameters."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (
            parameter
            for parameter in parameters
            if parameter.requires_grad
        )
    return sum(parameter.numel() for parameter in parameters)


def build_model(dataset_config):
    """Build the centralized model selected by ``dataset_config.name``.

    MNIST and FashionMNIST are currently supported. CIFAR-10 and IMDB will
    be added with their specialized paper architectures.
    """
    if dataset_config is None:
        raise ValueError("dataset_config must be provided.")

    dataset_name = _normalize_dataset_name(
        _config_get(dataset_config, "name")
    )
    num_classes = _config_get(dataset_config, "num_classes", 10)
    model_config = _config_get(dataset_config, "model", {}) or {}
    activation_config = (
        _config_get(model_config, "activation", {}) or {}
    )
    activation_parameters = {
        "scale": _config_get(activation_config, "scale", 2.0),
        "inverse_temperature": _config_get(
            activation_config,
            "inverse_temperature",
            2.0,
        ),
        "offset": _config_get(activation_config, "offset", 1.0),
    }

    if dataset_name == "mnist":
        model = MNISTCNN(
            num_classes=num_classes,
            activation_parameters=activation_parameters,
        )
        expected_parameters = 26_010
    else:
        hidden_width = _config_get(model_config, "hidden_width", 120)
        model = FashionMNISTMLP(
            num_classes=num_classes,
            hidden_width=hidden_width,
            activation_parameters=activation_parameters,
        )
        expected_parameters = 109_930

    if int(num_classes) == 10 and (
        dataset_name == "mnist" or int(hidden_width) == 120
    ):
        actual_parameters = count_parameters(model)
        if actual_parameters != expected_parameters:
            raise RuntimeError(
                f"Unexpected {dataset_name} parameter count: "
                f"expected {expected_parameters}, got "
                f"{actual_parameters}."
            )
    return model


def _config_get(config, key, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize_dataset_name(name):
    if name is None:
        raise ValueError("dataset_config.name must be defined.")
    normalized_name = str(name).strip().lower().replace("-", "_")
    try:
        return _DATASET_ALIASES[normalized_name]
    except KeyError as error:
        supported = ", ".join(sorted(set(_DATASET_ALIASES.values())))
        raise ValueError(
            f"No centralized model is implemented for dataset {name!r}. "
            f"Currently supported datasets: {supported}."
        ) from error


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    try:
        parsed_value = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{name} must be a positive integer; got {value!r}."
        ) from error
    if parsed_value < 1 or float(value) != parsed_value:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return parsed_value


def _finite_float(value, name):
    try:
        parsed_value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{name} must be a finite number; got {value!r}."
        ) from error
    if not math.isfinite(parsed_value):
        raise ValueError(
            f"{name} must be a finite number; got {value!r}."
        )
    return parsed_value


def _positive_finite_float(value, name):
    parsed_value = _finite_float(value, name)
    if parsed_value <= 0:
        raise ValueError(
            f"{name} must be positive; got {value!r}."
        )
    return parsed_value
