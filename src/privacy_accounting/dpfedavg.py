"""
Rényi privacy accounting for one complete DP-FedAvg training run.

The accounting hierarchy follows the existing project implementation:

1. Evaluate the Gaussian mechanism at integer Rényi orders.
2. Apply local data-subsampling amplification.
3. Compose across local updates.
4. Apply client-subsampling amplification for one communication round.
5. Compose across communication rounds.

Two local-subsampling accounting methods are supported:

``bounds``
    Use the integer-order subsampling upper bound at both the local-data
    and client-subsampling levels.

``numerical``
    Use ``dp_accounting`` for the composed Poisson-sampled Gaussian
    mechanism at the local-data level. The outer client-subsampling step
    still uses the same integer-order subsampling upper bound, because the
    inner mechanism is no longer a plain Gaussian mechanism.

The public function returns an :class:`rdp_utils.RdpCurve`. Conversion to
approximate DP belongs in ``rdp_utils.py`` and should be performed only
after all required RDP compositions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rdp_utils import (
    RdpCurve,
    apply_renyi_monotonicity_envelope,
    log1mexp,
)


AccountingMethod = Literal["bounds", "numerical"]
FloatArray = NDArray[np.float64]


def _validate_positive_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if value < 1:
        raise ValueError(f"{name} must be at least 1.")

    return value


def _validate_probability(value: float, name: str) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must satisfy 0 < {name} <= 1.")

    return value


def _validate_positive_float(value: float, name: str) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")

    return value


def _extract_mapping_value(
    mapping: Mapping[str, Any],
    *,
    canonical_name: str,
    aliases: tuple[str, ...],
) -> Any:
    """
    Extract one configuration value while supporting legacy aliases.
    """
    present = [
        key
        for key in (canonical_name, *aliases)
        if key in mapping
    ]

    if not present:
        accepted = ", ".join((canonical_name, *aliases))
        raise KeyError(
            f"Missing configuration value for {canonical_name!r}. "
            f"Accepted keys: {accepted}."
        )

    value = mapping[present[0]]

    for key in present[1:]:
        other_value = mapping[key]
        if other_value != value:
            raise ValueError(
                f"Conflicting values were supplied for {canonical_name!r}: "
                f"{present[0]}={value!r}, {key}={other_value!r}."
            )

    return value


@dataclass(frozen=True, slots=True)
class DPFedAvgConfig:
    """
    Configuration for one DP-FedAvg training run.

    Parameters
    ----------
    num_rounds:
        Number of communication rounds.

    num_local_updates:
        Number of privatized local updates composed for each participating
        client in one communication round.

    num_clients:
        Total number of clients.

    client_sampling_rate:
        Client-subsampling probability, denoted ``ell`` in the project code.

    local_sampling_rate:
        Local data-subsampling probability, denoted ``s`` in the project code.

    sigma_gaussian:
        Gaussian noise parameter.

    sigma_is_actual:
        If False, preserve the convention used in the existing project code:

            sigma_actual = sigma_gaussian
                           * sqrt(client_sampling_rate * num_clients).

        If True, ``sigma_gaussian`` is already the actual Gaussian noise
        multiplier supplied to the Gaussian mechanism.
    """

    num_rounds: int
    num_local_updates: int
    num_clients: int
    client_sampling_rate: float
    local_sampling_rate: float
    sigma_gaussian: float
    sigma_is_actual: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "num_rounds",
            _validate_positive_integer(
                self.num_rounds,
                "num_rounds",
            ),
        )
        object.__setattr__(
            self,
            "num_local_updates",
            _validate_positive_integer(
                self.num_local_updates,
                "num_local_updates",
            ),
        )
        object.__setattr__(
            self,
            "num_clients",
            _validate_positive_integer(
                self.num_clients,
                "num_clients",
            ),
        )
        object.__setattr__(
            self,
            "client_sampling_rate",
            _validate_probability(
                self.client_sampling_rate,
                "client_sampling_rate",
            ),
        )
        object.__setattr__(
            self,
            "local_sampling_rate",
            _validate_probability(
                self.local_sampling_rate,
                "local_sampling_rate",
            ),
        )
        object.__setattr__(
            self,
            "sigma_gaussian",
            _validate_positive_float(
                self.sigma_gaussian,
                "sigma_gaussian",
            ),
        )
        object.__setattr__(
            self,
            "sigma_is_actual",
            bool(self.sigma_is_actual),
        )

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
    ) -> "DPFedAvgConfig":
        """
        Construct a configuration from descriptive or legacy keys.

        Descriptive keys
        ----------------
        ``num_rounds``, ``num_local_updates``, ``num_clients``,
        ``client_sampling_rate``, ``local_sampling_rate``,
        ``sigma_gaussian``, and optionally ``sigma_is_actual``.

        Legacy keys
        -----------
        ``T``, ``K``, ``M``, ``l``, ``s``, and ``sigma_gaussian``.

        Additional keys, such as ``R``, are ignored because they do not
        affect the RDP curve of one training run.
        """
        return cls(
            num_rounds=_extract_mapping_value(
                config,
                canonical_name="num_rounds",
                aliases=("T",),
            ),
            num_local_updates=_extract_mapping_value(
                config,
                canonical_name="num_local_updates",
                aliases=("K",),
            ),
            num_clients=_extract_mapping_value(
                config,
                canonical_name="num_clients",
                aliases=("M",),
            ),
            client_sampling_rate=_extract_mapping_value(
                config,
                canonical_name="client_sampling_rate",
                aliases=("l", "ell"),
            ),
            local_sampling_rate=_extract_mapping_value(
                config,
                canonical_name="local_sampling_rate",
                aliases=("s",),
            ),
            sigma_gaussian=_extract_mapping_value(
                config,
                canonical_name="sigma_gaussian",
                aliases=("noise_multiplier",),
            ),
            sigma_is_actual=bool(
                config.get("sigma_is_actual", False)
            ),
        )

    @property
    def sigma_gaussian_actual(self) -> float:
        """
        Return the Gaussian noise multiplier used by the mechanism.
        """
        if self.sigma_is_actual:
            return self.sigma_gaussian

        return self.sigma_gaussian * math.sqrt(
            self.client_sampling_rate * self.num_clients
        )


def _normalize_config(
    config: DPFedAvgConfig | Mapping[str, Any],
) -> DPFedAvgConfig:
    if isinstance(config, DPFedAvgConfig):
        return config

    if isinstance(config, Mapping):
        return DPFedAvgConfig.from_mapping(config)

    raise TypeError(
        "config must be a DPFedAvgConfig or a mapping."
    )


def _normalize_accounting_method(
    accounting_method: str,
) -> AccountingMethod:
    normalized = accounting_method.strip().lower()

    aliases = {
        "bound": "bounds",
        "bounds": "bounds",
        "theory": "bounds",
        "theoretical": "bounds",
        "numerical": "numerical",
        "dp_accounting": "numerical",
    }

    if normalized not in aliases:
        raise ValueError(
            "accounting_method must be one of "
            "{'bounds', 'numerical'}."
        )

    return aliases[normalized]  # type: ignore[return-value]


def _normalize_requested_integer_orders(
    orders: ArrayLike,
) -> NDArray[np.int64]:
    """
    Validate requested integer Rényi orders.

    The accountant internally evaluates every integer order from 2 through
    the largest requested order, so sparse requested order sets are allowed.
    """
    orders_array = np.asarray(orders, dtype=float)

    if orders_array.ndim != 1:
        raise ValueError("orders must be one-dimensional.")
    if orders_array.size == 0:
        raise ValueError("orders must contain at least one order.")
    if not np.all(np.isfinite(orders_array)):
        raise ValueError("orders must contain only finite values.")
    if np.any(orders_array <= 1.0):
        raise ValueError(
            "Every requested Rényi order must be greater than 1."
        )
    if not np.all(
        np.isclose(
            orders_array,
            np.rint(orders_array),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "DP-FedAvg accounting currently supports integer Rényi "
            "orders only."
        )

    integer_orders = np.rint(orders_array).astype(np.int64)

    if np.any(np.diff(integer_orders) <= 0):
        raise ValueError(
            "Requested Rényi orders must be strictly increasing."
        )

    return integer_orders


def _log_combination(n: int, k: int) -> float:
    """
    Compute log(binomial(n, k)).
    """
    if k < 0 or k > n:
        return -math.inf

    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
    )


def _gaussian_rdp(
    order: int,
    sigma_gaussian_actual: float,
) -> float:
    """
    RDP epsilon of the Gaussian mechanism.
    """
    return (
        0.5
        * float(order)
        / (sigma_gaussian_actual ** 2)
    )


def _subsampled_cgf_bound_integer_order(
    order: int,
    *,
    sampling_probability: float,
    base_rdp_at_order,
) -> float:
    r"""
    Bound the CGF of a subsampled mechanism at an integer order.

    The return value is

        (order - 1) * epsilon_subsampled(order).

    ``base_rdp_at_order(j)`` must provide a valid RDP upper bound for the
    unsubsampled base mechanism at each integer ``j`` in {2, ..., order}.
    """
    order = _validate_positive_integer(order, "order")

    if order < 2:
        raise ValueError("order must be at least 2.")

    sampling_probability = _validate_probability(
        sampling_probability,
        "sampling_probability",
    )

    log_sampling_probability = math.log(
        sampling_probability
    )

    epsilon_two = float(base_rdp_at_order(2))

    if epsilon_two < 0.0 or not math.isfinite(epsilon_two):
        raise ValueError(
            "The base RDP bound at order 2 must be finite and "
            "nonnegative."
        )

    second_order_correction = min(
        math.log(4.0)
        + epsilon_two
        + log1mexp(epsilon_two),
        epsilon_two + math.log(2.0),
    )

    log_moment_two = (
        2.0 * log_sampling_probability
        + _log_combination(order, 2)
        + second_order_correction
    )

    log_terms = [0.0, log_moment_two]

    for j in range(3, order + 1):
        epsilon_j = float(base_rdp_at_order(j))

        if epsilon_j < 0.0 or not math.isfinite(epsilon_j):
            raise ValueError(
                f"The base RDP bound at order {j} must be finite "
                "and nonnegative."
            )

        log_terms.append(
            math.log(2.0)
            + (j - 1.0) * epsilon_j
            + j * log_sampling_probability
            + _log_combination(order, j)
        )

    return float(
        np.logaddexp.reduce(
            np.asarray(log_terms, dtype=float)
        )
    )


def _compute_local_rdp_with_bounds(
    dense_orders: NDArray[np.int64],
    config: DPFedAvgConfig,
) -> FloatArray:
    """
    Account for composed locally subsampled Gaussian updates using bounds.
    """
    sigma_actual = config.sigma_gaussian_actual
    local_sampling_rate = config.local_sampling_rate
    num_local_updates = config.num_local_updates

    def gaussian_rdp_at_order(order: int) -> float:
        return _gaussian_rdp(order, sigma_actual)

    local_rdp_values = []

    for order in dense_orders:
        order_int = int(order)

        subsampled_single_update_rdp = (
            _subsampled_cgf_bound_integer_order(
                order_int,
                sampling_probability=local_sampling_rate,
                base_rdp_at_order=gaussian_rdp_at_order,
            )
            / (order_int - 1.0)
        )

        unsubsampled_single_update_rdp = (
            gaussian_rdp_at_order(order_int)
        )

        local_rdp_values.append(
            num_local_updates
            * min(
                subsampled_single_update_rdp,
                unsubsampled_single_update_rdp,
            )
        )

    return np.asarray(local_rdp_values, dtype=float)


def _compute_local_rdp_numerically(
    dense_orders: NDArray[np.int64],
    config: DPFedAvgConfig,
) -> FloatArray:
    """
    Account for composed locally subsampled Gaussian updates with
    ``dp_accounting``.

    The unsubsampled Gaussian composition is also evaluated, and the
    pointwise minimum of the two valid upper bounds is returned.
    """
    try:
        from dp_accounting import dp_event
        from dp_accounting.rdp import RdpAccountant
    except ImportError as exc:
        raise ImportError(
            "accounting_method='numerical' requires the "
            "'dp-accounting' package."
        ) from exc

    orders_list = [
        int(order)
        for order in dense_orders
    ]

    single_step_event = dp_event.PoissonSampledDpEvent(
        sampling_probability=config.local_sampling_rate,
        event=dp_event.GaussianDpEvent(
            noise_multiplier=config.sigma_gaussian_actual,
        ),
    )

    total_local_event = dp_event.SelfComposedDpEvent(
        single_step_event,
        config.num_local_updates,
    )

    accountant = RdpAccountant(orders_list)
    accountant.compose(total_local_event)

    # dp_accounting currently stores the evaluated curve in _rdp.
    numerical_values = np.asarray(
        accountant._rdp,
        dtype=float,
    )

    unsubsampled_values = np.asarray(
        [
            config.num_local_updates
            * _gaussian_rdp(
                int(order),
                config.sigma_gaussian_actual,
            )
            for order in dense_orders
        ],
        dtype=float,
    )

    return np.minimum(
        numerical_values,
        unsubsampled_values,
    )


def _compute_full_training_rdp_dense(
    dense_orders: NDArray[np.int64],
    config: DPFedAvgConfig,
    accounting_method: AccountingMethod,
) -> FloatArray:
    """
    Compute the full DP-FedAvg RDP curve on dense integer orders.
    """
    if accounting_method == "bounds":
        local_rdp_values = _compute_local_rdp_with_bounds(
            dense_orders,
            config,
        )
    else:
        local_rdp_values = _compute_local_rdp_numerically(
            dense_orders,
            config,
        )

    local_rdp_by_order = {
        int(order): float(epsilon)
        for order, epsilon in zip(
            dense_orders,
            local_rdp_values,
        )
    }

    def local_rdp_at_order(order: int) -> float:
        try:
            return local_rdp_by_order[int(order)]
        except KeyError as exc:
            raise KeyError(
                f"Missing local RDP value at integer order {order}."
            ) from exc

    full_training_rdp = []

    for order in dense_orders:
        order_int = int(order)

        client_subsampled_round_rdp = (
            _subsampled_cgf_bound_integer_order(
                order_int,
                sampling_probability=config.client_sampling_rate,
                base_rdp_at_order=local_rdp_at_order,
            )
            / (order_int - 1.0)
        )

        unsubsampled_round_rdp = local_rdp_at_order(
            order_int
        )

        per_round_rdp = min(
            client_subsampled_round_rdp,
            unsubsampled_round_rdp,
        )

        full_training_rdp.append(
            config.num_rounds * per_round_rdp
        )

    return np.asarray(
        full_training_rdp,
        dtype=float,
    )


def compute_dpfedavg_rdp(
    config: DPFedAvgConfig | Mapping[str, Any],
    orders: ArrayLike,
    *,
    accounting_method: str = "bounds",
    apply_monotonicity: bool = True,
) -> RdpCurve:
    """
    Compute the RDP curve of one complete DP-FedAvg training run.

    Parameters
    ----------
    config:
        A :class:`DPFedAvgConfig` or a compatible mapping. Legacy project
        dictionaries using keys ``T``, ``K``, ``M``, ``l``, ``s``, and
        ``sigma_gaussian`` are accepted.

    orders:
        Strictly increasing integer Rényi orders greater than 1.

    accounting_method:
        ``"bounds"`` or ``"numerical"``.

        - ``"bounds"`` uses the integer-order subsampling bound for the
          local-data and client-subsampling levels.
        - ``"numerical"`` uses ``dp_accounting`` for the locally
          Poisson-sampled Gaussian composition and the same theorem bound
          for client subsampling.

    apply_monotonicity:
        Whether to improve the final dense RDP upper-bound curve using
        Rényi-order monotonicity before selecting the requested orders.

    Returns
    -------
    RdpCurve
        RDP upper bounds at exactly the requested orders.
    """
    normalized_config = _normalize_config(config)
    normalized_method = _normalize_accounting_method(
        accounting_method
    )
    requested_orders = _normalize_requested_integer_orders(
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
        epsilons=_compute_full_training_rdp_dense(
            dense_orders,
            normalized_config,
            normalized_method,
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
