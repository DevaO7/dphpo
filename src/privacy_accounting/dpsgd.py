"""Rényi privacy accounting for one central DP-SGD training run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config_utils import extract_mapping_value
from .rdp_utils import (
    RdpCurve,
    apply_renyi_monotonicity_envelope,
    gaussian_rdp,
    normalize_requested_integer_orders,
)
from .validation import (
    validate_positive_float,
    validate_positive_integer,
    validate_probability,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DPSGDConfig:
    """
    Configuration for one DP-SGD training run.

    Parameters
    ----------
    num_rounds:
        Number of optimizer updates (central training iterations) to compose.

    data_sampling_rate:
        Poisson data-subsampling probability for each optimizer update.

    sigma_gaussian:
        Gaussian noise multiplier used by DP-SGD.
    """

    num_rounds: int
    data_sampling_rate: float
    sigma_gaussian: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "num_rounds",
            validate_positive_integer(
                self.num_rounds,
                "num_rounds",
            ),
        )
        object.__setattr__(
            self,
            "data_sampling_rate",
            validate_probability(
                self.data_sampling_rate,
                "data_sampling_rate",
            ),
        )
        object.__setattr__(
            self,
            "sigma_gaussian",
            validate_positive_float(
                self.sigma_gaussian,
                "sigma_gaussian",
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
    ) -> "DPSGDConfig":
        """
        Construct a configuration from descriptive or legacy keys.

        Accepted canonical keys are ``num_rounds``,
        ``data_sampling_rate``, and ``sigma_gaussian``. The aliases ``T``,
        ``sampling_rate``, ``sample_rate``, ``s``, and
        ``noise_multiplier`` support the experiment and legacy schemas.
        Unrelated mapping entries are ignored.
        """
        return cls(
            num_rounds=extract_mapping_value(
                config,
                canonical_name="num_rounds",
                aliases=("T",),
            ),
            data_sampling_rate=extract_mapping_value(
                config,
                canonical_name="data_sampling_rate",
                aliases=("sampling_rate", "sample_rate", "s"),
            ),
            sigma_gaussian=extract_mapping_value(
                config,
                canonical_name="sigma_gaussian",
                aliases=("noise_multiplier",),
            ),
        )


def _compute_dpsgd_rdp_numerically(
    dense_orders: NDArray[np.int64],
    config: DPSGDConfig,
) -> FloatArray:
    """
    Account for composed Poisson-subsampled Gaussian optimizer updates.

    The unsubsampled Gaussian composition is also evaluated, and the
    pointwise minimum of the two valid upper bounds is returned.
    """
    try:
        from dp_accounting import dp_event
        from dp_accounting.rdp import RdpAccountant
    except ImportError as exc:
        raise ImportError(
            "DP-SGD accounting requires the 'dp-accounting' package."
        ) from exc

    orders_list = [
        int(order)
        for order in dense_orders
    ]

    single_step_event = dp_event.PoissonSampledDpEvent(
        sampling_probability=config.data_sampling_rate,
        event=dp_event.GaussianDpEvent(
            noise_multiplier=config.sigma_gaussian,
        ),
    )

    total_training_event = dp_event.SelfComposedDpEvent(
        single_step_event,
        config.num_rounds,
    )

    accountant = RdpAccountant(orders_list)
    accountant.compose(total_training_event)

    # Newer dp_accounting versions expose the curve publicly. Retain the
    # private-attribute fallback for older supported installations.
    accountant_rdp = getattr(accountant, "rdp", None)
    if accountant_rdp is None:
        accountant_rdp = accountant._rdp
    numerical_values = np.asarray(accountant_rdp, dtype=float)

    unsubsampled_values = np.asarray(
        [
            config.num_rounds
            * gaussian_rdp(
                int(order),
                config.sigma_gaussian,
            )
            for order in dense_orders
        ],
        dtype=float,
    )

    return np.minimum(
        numerical_values,
        unsubsampled_values,
    )


def _normalize_config(
    config: DPSGDConfig | Mapping[str, Any],
) -> DPSGDConfig:
    if isinstance(config, DPSGDConfig):
        return config

    if isinstance(config, Mapping):
        return DPSGDConfig.from_mapping(config)

    raise TypeError(
        "config must be a DPSGDConfig or a mapping."
    )


def compute_dpsgd_rdp(
    config: DPSGDConfig | Mapping[str, Any],
    orders: ArrayLike,
    *,
    apply_monotonicity: bool = True,
) -> RdpCurve:
    """Compute the RDP curve of a central DP-SGD training segment.

    The accountant composes ``num_rounds`` Poisson-sampled Gaussian
    optimizer updates. For a resumed second stage, ``num_rounds`` must be
    that stage's incremental number of updates, not its cumulative endpoint.
    """
    normalized_config = _normalize_config(config)
    requested_orders = normalize_requested_integer_orders(
        orders
    )
    max_order = int(requested_orders[-1])
    dense_orders = np.arange(
        2,
        max_order + 1,
        dtype=np.int64,
    )
    dense_curve = RdpCurve(
        orders=dense_orders.astype(float),
        epsilons=_compute_dpsgd_rdp_numerically(
            dense_orders,
            normalized_config,
        ),
    )
    if apply_monotonicity:
        dense_curve = apply_renyi_monotonicity_envelope(
            dense_curve
        )

    requested_indices = requested_orders - 2

    return RdpCurve(
        orders=requested_orders.astype(float),
        epsilons=dense_curve.epsilons[requested_indices],
    )
