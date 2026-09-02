import csv
from dataclasses import dataclass
import math
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from omegaconf import DictConfig
from scipy.optimize import brentq
from scipy.special import gammainc

from privacy_accounting.dpsgd import compute_dpsgd_rdp
from privacy_accounting.poisson import PoissonDistribution
from privacy_accounting.rdp_utils import (
    ApproxDPResult,
    RdpCurve,
    compose_rdp_curves,
    convert_rdp_to_approx_dp,
)
from privacy_accounting.selection_accounting import (
    PoissonNStageResult,
    PoissonTop1Result,
    compute_top1_rdp,
    compute_top1_rdp_poisson,
    compute_top_m_rdp,
    compute_n_stage_rdp_poisson,
)


def _coverage_probability_conditioned_on_large_m(
    *,
    mu,
    num_configurations,
    m,
    relative_tolerance=1e-13,
):
    """Evaluate coverage under ``Poisson(mu) | K >= m`` for ``m > N``.

    The computation sums over the conditioned Poisson trial count. For each
    count, a stable occupancy recurrence gives the probability that all
    configurations have appeared; this avoids cancellation in the paper's
    inclusion-exclusion sum.
    """
    distribution = PoissonDistribution(mu)
    conditioning_probability = distribution.survival_probability(m)

    # occupied_probabilities[j] is the probability that exactly j of the N
    # configurations have appeared after the current number of draws.
    occupied_probabilities = np.zeros(num_configurations + 1, dtype=float)
    occupied_probabilities[0] = 1.0
    occupied_indices = np.arange(num_configurations + 1, dtype=float)
    newly_occupied_factors = (
        num_configurations - np.arange(num_configurations, dtype=float)
    ) / num_configurations

    alpha_m = 0.0
    k = 0
    minimum_tail_check = max(m, int(math.ceil(mu)))
    maximum_k = minimum_tail_check + 1_000_000
    while k < maximum_k:
        k += 1
        next_probabilities = (
            occupied_probabilities
            * occupied_indices
            / num_configurations
        )
        next_probabilities[1:] += (
            occupied_probabilities[:-1] * newly_occupied_factors
        )
        occupied_probabilities = next_probabilities

        if k < m:
            continue

        log_conditional_probability = (
            -mu
            + k * math.log(mu)
            - math.lgamma(k + 1)
            - math.log(conditioning_probability)
        )
        alpha_m += (
            math.exp(log_conditional_probability)
            * occupied_probabilities[num_configurations]
        )

        if k >= minimum_tail_check:
            # P[K > k | K >= m] bounds the contribution of all omitted
            # terms because every occupancy probability is at most one.
            remaining_conditional_probability = (
                float(gammainc(k + 1, mu))
                / conditioning_probability
            )
            relative_scale = max(alpha_m, np.finfo(float).tiny)
            if remaining_conditional_probability <= (
                relative_tolerance * relative_scale
            ):
                return alpha_m

    raise ArithmeticError(
        "Conditioned coverage summation did not converge within "
        f"{maximum_k} Poisson counts."
    )


def compute_coverage_probabilities(
    mu_values,
    *,
    num_configurations,
    m,
):
    r"""Evaluate the paper's :math:`\alpha_P` and :math:`\alpha_m`.

    The top-m result conditions a Poisson trial count on ``K >= m``.
    When ``m <= N``, observing all ``N`` configurations implies the
    conditioning event, so the infinite sum in the paper simplifies to

        alpha_m = alpha_P / P[Poisson(mu) >= m].

    When ``m > N``, repeated configurations are still allowed and coverage
    remains possible. In that case the full conditioned expectation is
    evaluated with a stable occupancy recurrence.
    """
    if (
        not isinstance(num_configurations, (int, np.integer))
        or isinstance(num_configurations, bool)
        or num_configurations <= 0
    ):
        raise ValueError(
            "num_configurations must be a positive integer."
        )
    if (
        not isinstance(m, (int, np.integer))
        or isinstance(m, bool)
        or m <= 0
    ):
        raise ValueError("m must be a positive integer.")

    num_configurations = int(num_configurations)
    m = int(m)
    mu_values = np.asarray(mu_values, dtype=float)
    if mu_values.ndim != 1 or mu_values.size == 0:
        raise ValueError("mu_values must be a non-empty 1-D sequence.")
    if not np.all(np.isfinite(mu_values)) or np.any(mu_values <= 0.0):
        raise ValueError(
            "Every Poisson mean in mu_values must be finite and positive."
        )

    probability_one_or_more_per_configuration = -np.expm1(
        -mu_values / num_configurations
    )
    alpha_p = np.power(
        probability_one_or_more_per_configuration,
        num_configurations,
    )
    if m <= num_configurations:
        conditioning_probabilities = np.asarray(
            [
                PoissonDistribution(mu).survival_probability(m)
                for mu in mu_values
            ],
            dtype=float,
        )
        alpha_m = alpha_p / conditioning_probabilities
    else:
        alpha_m = np.asarray(
            [
                _coverage_probability_conditioned_on_large_m(
                    mu=float(mu),
                    num_configurations=num_configurations,
                    m=m,
                )
                for mu in mu_values
            ],
            dtype=float,
        )

    tolerance = 1e-12
    for name, probabilities in (
        ("alpha_p", alpha_p),
        ("alpha_m", alpha_m),
    ):
        if not np.all(np.isfinite(probabilities)):
            raise ArithmeticError(f"{name} contains a non-finite value.")
        if np.any(probabilities < -tolerance) or np.any(
            probabilities > 1.0 + tolerance
        ):
            raise ArithmeticError(
                f"{name} contains a value outside [0, 1]."
            )
    if np.any(alpha_m + tolerance < alpha_p):
        raise ArithmeticError(
            "Conditioning on K >= m unexpectedly reduced coverage."
        )

    return np.clip(alpha_p, 0.0, 1.0), np.clip(alpha_m, 0.0, 1.0)


def _fixed_trial_coverage_probability(*, num_trials, num_configurations):
    """Return the probability of covering all configurations in k draws."""
    if num_trials < num_configurations:
        return 0.0

    occupied_probabilities = np.zeros(num_configurations + 1, dtype=float)
    occupied_probabilities[0] = 1.0
    occupied_indices = np.arange(num_configurations + 1, dtype=float)
    newly_occupied_factors = (
        num_configurations - np.arange(num_configurations, dtype=float)
    ) / num_configurations
    for _ in range(num_trials):
        next_probabilities = (
            occupied_probabilities
            * occupied_indices
            / num_configurations
        )
        next_probabilities[1:] += (
            occupied_probabilities[:-1] * newly_occupied_factors
        )
        occupied_probabilities = next_probabilities
    return float(occupied_probabilities[num_configurations])


def solve_poisson_means_for_coverage(
    target_probability,
    *,
    num_configurations,
    m,
):
    """Find underlying Poisson rates achieving a coverage target.

    Returns the Papernot rate and, when attainable at a finite positive rate,
    the conditioned top-m rate. The top-m limiting coverage as ``mu -> 0+``
    is also returned so unattainable targets are explicit.
    """
    if (
        not isinstance(num_configurations, (int, np.integer))
        or isinstance(num_configurations, bool)
        or num_configurations <= 0
    ):
        raise ValueError(
            "num_configurations must be a positive integer."
        )
    if (
        not isinstance(m, (int, np.integer))
        or isinstance(m, bool)
        or m <= 0
    ):
        raise ValueError("m must be a positive integer.")
    target_probability = float(target_probability)
    if (
        not math.isfinite(target_probability)
        or not 0.0 < target_probability < 1.0
    ):
        raise ValueError(
            "target_probability must be finite and strictly between 0 and 1."
        )

    num_configurations = int(num_configurations)
    m = int(m)

    # Invert alpha_P = (1 - exp(-mu / N))**N using stable primitives.
    probability_per_configuration = math.exp(
        math.log(target_probability) / num_configurations
    )
    papernot_mu = -num_configurations * math.log1p(
        -probability_per_configuration
    )

    minimum_top_m_coverage = _fixed_trial_coverage_probability(
        num_trials=m,
        num_configurations=num_configurations,
    )
    comparison_tolerance = 1e-14
    if target_probability < (
        minimum_top_m_coverage - comparison_tolerance
    ):
        return {
            "target_probability": target_probability,
            "papernot_mu": papernot_mu,
            "top_m_mu": None,
            "minimum_top_m_coverage": minimum_top_m_coverage,
            "top_m_status": "below_mu_to_zero_limit",
        }
    if math.isclose(
        target_probability,
        minimum_top_m_coverage,
        rel_tol=0.0,
        abs_tol=comparison_tolerance,
    ):
        return {
            "target_probability": target_probability,
            "papernot_mu": papernot_mu,
            "top_m_mu": None,
            "minimum_top_m_coverage": minimum_top_m_coverage,
            "top_m_status": "attained_only_as_mu_tends_to_zero",
        }

    def top_m_objective(mu):
        return float(
            compute_coverage_probabilities(
                [mu],
                num_configurations=num_configurations,
                m=m,
            )[1][0]
            - target_probability
        )

    # Conditioning can only increase coverage at a fixed Poisson rate, so
    # the unconditioned Papernot solution is a valid upper bracket.
    upper_mu = papernot_mu
    upper_value = top_m_objective(upper_mu)
    root_value_tolerance = 1e-12
    if abs(upper_value) <= root_value_tolerance:
        return {
            "target_probability": target_probability,
            "papernot_mu": papernot_mu,
            "top_m_mu": upper_mu,
            "minimum_top_m_coverage": minimum_top_m_coverage,
            "top_m_status": "solved_at_numerical_precision",
        }
    if upper_value < -comparison_tolerance:
        raise ArithmeticError(
            "Could not bracket the top-m coverage rate from above."
        )

    lower_mu = min(1.0, upper_mu / 2.0)
    lower_value = top_m_objective(lower_mu)
    for _ in range(200):
        if lower_value < 0.0:
            break
        lower_mu /= 2.0
        lower_value = top_m_objective(lower_mu)
    else:
        raise ArithmeticError(
            "Could not numerically bracket the top-m coverage rate near "
            "its mu -> 0 limit."
        )

    top_m_mu = float(
        brentq(
            top_m_objective,
            lower_mu,
            upper_mu,
            xtol=1e-12,
            rtol=1e-12,
            maxiter=300,
        )
    )
    return {
        "target_probability": target_probability,
        "papernot_mu": papernot_mu,
        "top_m_mu": top_m_mu,
        "minimum_top_m_coverage": minimum_top_m_coverage,
        "top_m_status": "solved",
    }


def _build_mu_values(mu_config, *, field_name="coverage.mu"):
    mu_min = float(mu_config.min)
    mu_max = float(mu_config.max)
    num_points = int(mu_config.num_points)
    scale = str(mu_config.scale).strip().lower()

    if not math.isfinite(mu_min) or mu_min <= 0.0:
        raise ValueError(f"{field_name}.min must be finite and positive.")
    if not math.isfinite(mu_max) or mu_max <= mu_min:
        raise ValueError(
            f"{field_name}.max must be finite and greater than its min."
        )
    if num_points < 2:
        raise ValueError(f"{field_name}.num_points must be at least 2.")

    if scale == "linear":
        return np.linspace(mu_min, mu_max, num_points), scale
    if scale == "log":
        return (
            np.logspace(
                math.log10(mu_min),
                math.log10(mu_max),
                num_points,
            ),
            scale,
        )
    raise ValueError(f"{field_name}.scale must be 'linear' or 'log'.")


def coverage_probability_vs_mean_plot(config: DictConfig):
    """Plot Papernot and conditioned top-m coverage versus Poisson mean."""
    exp_config = config.experiment
    coverage_config = exp_config.coverage
    num_configurations = coverage_config.num_configurations
    m = coverage_config.m
    mu_values, mu_scale = _build_mu_values(coverage_config.mu)
    alpha_p, alpha_m = compute_coverage_probabilities(
        mu_values,
        num_configurations=num_configurations,
        m=m,
    )
    num_configurations = int(num_configurations)
    m = int(m)

    figure_size = tuple(float(value) for value in exp_config.plot.figsize)
    if len(figure_size) != 2 or any(value <= 0.0 for value in figure_size):
        raise ValueError("plot.figsize must contain two positive values.")
    dpi = int(exp_config.plot.dpi)
    if dpi <= 0:
        raise ValueError("plot.dpi must be positive.")
    probability_scale = str(
        exp_config.plot.get("probability_scale", "linear")
    ).strip().lower()
    if probability_scale not in {"linear", "log"}:
        raise ValueError(
            "plot.probability_scale must be 'linear' or 'log'."
        )
    if probability_scale == "log" and (
        np.any(alpha_p <= 0.0) or np.any(alpha_m <= 0.0)
    ):
        raise ValueError(
            "The configured mu range underflows to zero probability; "
            "increase coverage.mu.min or use a linear probability scale."
        )

    output_directory = Path(HydraConfig.get().runtime.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    data_path = output_directory / str(exp_config.output.data_filename)
    with data_path.open(mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["mu", "alpha_p", "alpha_m"])
        writer.writerows(zip(mu_values, alpha_p, alpha_m))

    target_probabilities = [
        float(value)
        for value in exp_config.inverse_coverage.target_probabilities
    ]
    if not target_probabilities:
        raise ValueError(
            "inverse_coverage.target_probabilities must not be empty."
        )
    inverse_rows = [
        solve_poisson_means_for_coverage(
            target_probability,
            num_configurations=num_configurations,
            m=m,
        )
        for target_probability in target_probabilities
    ]
    inverse_data_path = output_directory / str(
        exp_config.inverse_coverage.output_filename
    )
    with inverse_data_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "target_probability",
            "papernot_mu",
            "top_m_mu",
            "minimum_top_m_coverage",
            "top_m_status",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inverse_rows)

    figure, axis = plt.subplots(figsize=figure_size)
    axis.plot(
        mu_values,
        alpha_p,
        label=r"Papernot ($\alpha_P$)",
        linewidth=2.0,
    )
    axis.plot(
        mu_values,
        alpha_m,
        label=rf"Top-{m}, $K \mid K\geq {m}$ ($\alpha_m$)",
        linewidth=2.0,
        linestyle="--",
    )
    axis.set_xscale(mu_scale)
    axis.set_yscale(probability_scale)
    if probability_scale == "linear":
        axis.set_ylim(-0.02, 1.02)
    else:
        minimum_probability = min(float(alpha_p[0]), float(alpha_m[0]))
        axis.set_ylim(minimum_probability / 2.0, 1.2)
    axis.set_xlabel(r"Underlying Poisson mean $\mu$")
    axis.set_ylabel("Probability all configurations are sampled")
    axis.set_title(
        rf"Coverage Probability ($N={num_configurations}$, $m={m}$)"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    plot_path = output_directory / str(exp_config.output.plot_filename)
    figure.savefig(plot_path, dpi=dpi)
    plt.close(figure)

    print(f"Saved coverage probabilities: {data_path}", flush=True)
    print(f"Saved inverse coverage results: {inverse_data_path}", flush=True)
    print(f"Saved coverage plot: {plot_path}", flush=True)
    return {
        "data_path": data_path,
        "inverse_data_path": inverse_data_path,
        "plot_path": plot_path,
    }


def compare_with_peeling_plot():
    """Compare direct top-m accounting with composed top-1 accounting.

    The composed top-1 curve is an accounting baseline. It does not by
    itself establish equivalence to sequentially removing winners from a
    single shared random pool of candidates.
    """
    orders = np.arange(2, 101)
    E_k_min = 3.05
    E_k_max = 1_000_000
    num_E_k_points = 120
    eta = 0.0
    m = 3
    delta = 1e-5

    # base mechanism
    base_mechanism = RdpCurve(
        orders=orders,
        epsilons=0.1 * orders,
    )

    E_k_values = np.logspace(
        math.log10(E_k_min),
        math.log10(E_k_max),
        num_E_k_points,
    )

    eps_topm_list = []
    eps_peeling_list = []
    for E_k in E_k_values:
        # papernot top-1
        peeling_rdp_curves = []
        for peel in range(m):
            papernot_top1_result = compute_top1_rdp(
                base_rdp_curve=base_mechanism,
                expected_num_trials=E_k - peel,
                eta=eta,
            )
            peeling_rdp_curves.append(
                papernot_top1_result.rdp_curve
            )

        peeling_composed = compose_rdp_curves(*peeling_rdp_curves)

        # top-m
        top_m_result = compute_top_m_rdp(
            base_rdp_curve=base_mechanism,
            m=m,
            expected_num_trials=E_k,
            eta=eta,
        )
        peeling_composed_dp = convert_rdp_to_approx_dp(
            peeling_composed,
            delta=delta,
        )
        top_m_dp = convert_rdp_to_approx_dp(
            top_m_result.rdp_curve,
            delta=delta,
        )
        eps_peeling_list.append(peeling_composed_dp.epsilon)
        eps_topm_list.append(top_m_dp.epsilon)

    plt.figure(figsize=(10, 6))
    plt.plot(
        E_k_values,
        eps_peeling_list,
        label="Composed top-1 baseline",
        marker="o",
    )
    plt.plot(
        E_k_values,
        eps_topm_list,
        label="Direct top-m",
        marker="x",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Expected number of trials $\mathbb{E}[K]$")
    plt.ylabel("Epsilon (log scale)")
    plt.title("Privacy Accounting: Peeling vs Top-M")
    plt.legend()
    plt.tight_layout()

    output_path = Path(
        "results/exp0_privacy_accounting/"
        "privacy_accounting_comparison.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


@dataclass(frozen=True, slots=True)
class FixedScheduleStage:
    """One predeclared rung in a fixed expected-compute schedule."""

    stage: int
    expected_num_trials: float
    retained_count: int
    cumulative_resource: int
    incremental_resource: int
    count_semantics: str


@dataclass(frozen=True, slots=True)
class FixedSchedulePrivacyResult:
    """Privacy and compute comparison for one fixed schedule."""

    schedule: tuple[FixedScheduleStage, ...]
    n_stage: PoissonNStageResult
    papernot: PoissonTop1Result
    n_stage_dp: ApproxDPResult
    papernot_dp: ApproxDPResult
    n_stage_expected_compute: float
    papernot_expected_compute: float


def _round_schedule_value(value, *, mode, name):
    """Round one positive schedule quantity using an explicit rule."""
    value = float(value)
    mode = str(mode).strip().lower()
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    if mode == "floor":
        rounded = math.floor(value)
    elif mode == "ceil":
        rounded = math.ceil(value)
    elif mode == "nearest":
        rounded = math.floor(value + 0.5)
    else:
        raise ValueError(
            f"{name} rounding mode must be 'floor', 'ceil', or 'nearest'."
        )
    return max(int(rounded), 1)


def build_fixed_poisson_schedule(
    *,
    initial_expected_num_trials,
    retention_factor,
    num_stages,
    minimum_resource,
    survivor_rounding="floor",
    resource_rounding="nearest",
):
    r"""Construct a geometric, data-independent DP-SHA schedule.

    For every stage ``l`` (one-indexed), let ``w_l`` be the intended
    expected run count. The schedule is

    .. math::

        w_l = \bar\mu_1 q^{l-1},\qquad
        r_l = r_{\min}q^{-(l-1)}.

    For ``l < L``, ``w_l`` is the conditional mean ``bar_mu_l``. For the
    unconditioned top-1 final stage, ``w_L`` is the Poisson rate
    ``theta_L``.

    Every non-final stage retains ``m_l = round(q * bar_mu_l)`` using the
    requested integer rule.  The final stage releases top-1.  The schedule
    is built before any trial count is sampled; no realized-compute
    recycling is performed.
    """
    initial_expected_num_trials = float(initial_expected_num_trials)
    retention_factor = float(retention_factor)
    if (
        not math.isfinite(initial_expected_num_trials)
        or initial_expected_num_trials <= 0.0
    ):
        raise ValueError(
            "initial_expected_num_trials must be finite and positive."
        )
    if (
        not math.isfinite(retention_factor)
        or not 0.0 < retention_factor < 1.0
    ):
        raise ValueError("retention_factor must satisfy 0 < q < 1.")
    if (
        not isinstance(num_stages, (int, np.integer))
        or isinstance(num_stages, bool)
        or num_stages <= 0
    ):
        raise ValueError("num_stages must be a positive integer.")
    if (
        not isinstance(minimum_resource, (int, np.integer))
        or isinstance(minimum_resource, bool)
        or minimum_resource <= 0
    ):
        raise ValueError("minimum_resource must be a positive integer.")

    stages = []
    previous_resource = 0
    previous_retained_count = None
    for stage_index in range(int(num_stages)):
        stage_number = stage_index + 1
        expected_num_trials = (
            initial_expected_num_trials
            * retention_factor**stage_index
        )
        cumulative_resource = _round_schedule_value(
            minimum_resource / retention_factor**stage_index,
            mode=resource_rounding,
            name="cumulative resource",
        )
        if cumulative_resource <= previous_resource:
            raise ValueError(
                "Rounded cumulative resources must strictly increase. "
                "Use a smaller retention factor, a larger minimum "
                "resource, or a different rounding rule."
            )

        is_final = stage_number == int(num_stages)
        if is_final:
            retained_count = 1
            count_semantics = "unconditioned_poisson_mean"
        else:
            retained_count = _round_schedule_value(
                retention_factor * expected_num_trials,
                mode=survivor_rounding,
                name="retained count",
            )
            count_semantics = "conditional_mean_given_k_ge_m"
            if expected_num_trials <= retained_count:
                raise ValueError(
                    f"Stage {stage_number} requires bar_mu_l > m_l for "
                    "finite conditioned-Poisson calibration, but received "
                    f"bar_mu_l={expected_num_trials:g} and "
                    f"m_l={retained_count}. Increase the initial breadth "
                    "or reduce the number of stages."
                )

        if (
            previous_retained_count is not None
            and retained_count > previous_retained_count
        ):
            raise ValueError(
                "The rounded retained counts must be nonincreasing across "
                "stages."
            )
        stages.append(
            FixedScheduleStage(
                stage=stage_number,
                expected_num_trials=float(expected_num_trials),
                retained_count=retained_count,
                cumulative_resource=cumulative_resource,
                incremental_resource=(
                    cumulative_resource - previous_resource
                ),
                count_semantics=count_semantics,
            )
        )
        previous_resource = cumulative_resource
        previous_retained_count = retained_count

    return tuple(stages)


def _compute_fixed_schedule_base_curves(
    schedule,
    *,
    sampling_rate,
    noise_multiplier,
    orders,
):
    """Account for each incremental training segment and the full run."""
    stage_base_curves = tuple(
        compute_dpsgd_rdp(
            config={
                "num_rounds": stage.incremental_resource,
                "data_sampling_rate": sampling_rate,
                "sigma_gaussian": noise_multiplier,
            },
            orders=orders,
        )
        for stage in schedule
    )
    papernot_base_curve = compute_dpsgd_rdp(
        config={
            "num_rounds": schedule[-1].cumulative_resource,
            "data_sampling_rate": sampling_rate,
            "sigma_gaussian": noise_multiplier,
        },
        orders=orders,
    )
    return stage_base_curves, papernot_base_curve


def compute_fixed_schedule_privacy(
    schedule,
    *,
    delta,
    stage_base_curves,
    papernot_base_curve,
):
    """Compare fixed N-stage accounting to breadth-matched Papernot."""
    schedule = tuple(schedule)
    stage_base_curves = tuple(stage_base_curves)
    if not schedule:
        raise ValueError("schedule must contain at least one stage.")
    if len(stage_base_curves) != len(schedule):
        raise ValueError(
            "stage_base_curves must contain one curve per schedule stage."
        )

    n_stage = compute_n_stage_rdp_poisson(
        stage_base_curves,
        retained_counts=[stage.retained_count for stage in schedule],
        expected_num_trials=[
            stage.expected_num_trials for stage in schedule
        ],
    )
    n_stage_dp = convert_rdp_to_approx_dp(
        n_stage.rdp_curve,
        delta=delta,
    )

    # Breadth matching equates Papernot's unconditioned expected count
    # theta_P with the first stage's conditional expected count bar_mu_1.
    papernot = compute_top1_rdp_poisson(
        papernot_base_curve,
        expected_num_trials=schedule[0].expected_num_trials,
    )
    papernot_dp = convert_rdp_to_approx_dp(
        papernot.rdp_curve,
        delta=delta,
    )

    for method_name, approximate_dp in (
        ("N-stage", n_stage_dp),
        ("breadth-matched Papernot", papernot_dp),
    ):
        if approximate_dp.is_at_max_order:
            raise ArithmeticError(
                f"The {method_name} optimum is at the maximum Renyi "
                "order. Increase privacy.max_renyi_order before using "
                "this result."
            )

    n_stage_expected_compute = sum(
        stage.expected_num_trials * stage.incremental_resource
        for stage in schedule
    )
    papernot_expected_compute = (
        schedule[0].expected_num_trials
        * schedule[-1].cumulative_resource
    )
    return FixedSchedulePrivacyResult(
        schedule=schedule,
        n_stage=n_stage,
        papernot=papernot,
        n_stage_dp=n_stage_dp,
        papernot_dp=papernot_dp,
        n_stage_expected_compute=float(n_stage_expected_compute),
        papernot_expected_compute=float(papernot_expected_compute),
    )


def _write_dict_rows(path, *, fieldnames, rows):
    path = Path(path)
    with path.open(mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_plot_settings(plot_config):
    figure_size = tuple(float(value) for value in plot_config.figsize)
    if len(figure_size) != 2 or any(value <= 0.0 for value in figure_size):
        raise ValueError("plot.figsize must contain two positive values.")
    dpi = int(plot_config.dpi)
    if dpi <= 0:
        raise ValueError("plot.dpi must be positive.")
    x_scale = str(plot_config.get("x_scale", "linear")).strip().lower()
    if x_scale not in {"linear", "log"}:
        raise ValueError("plot.x_scale must be 'linear' or 'log'.")
    return figure_size, dpi, x_scale


def _plot_accounting_series(
    *,
    rows,
    noise_multipliers,
    figure_size,
    dpi,
    x_scale,
    output_directory,
    output_config,
):
    """Create privacy, privacy-gap, and expected-compute figures."""
    plot_specs = (
        (
            "epsilon_plot_filename",
            "epsilon_n_stage",
            "epsilon_papernot_breadth_matched",
            r"Privacy $\epsilon$",
            "Fixed-Schedule Privacy Accounting",
        ),
        (
            "expected_compute_plot_filename",
            "expected_compute_n_stage",
            "expected_compute_papernot_breadth_matched",
            "Expected communication rounds",
            "Expected Compute of the Compared Schedules",
        ),
    )
    plot_paths = {}
    for (
        filename_key,
        n_stage_key,
        papernot_key,
        y_label,
        title,
    ) in plot_specs:
        figure, axis = plt.subplots(figsize=figure_size)
        for noise_multiplier in noise_multipliers:
            sigma_rows = [
                row
                for row in rows
                if math.isclose(
                    row["noise_multiplier"],
                    noise_multiplier,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ]
            x_values = [
                row["initial_conditional_expected_runs_bar_mu_1"]
                for row in sigma_rows
            ]
            suffix = (
                "" if len(noise_multipliers) == 1
                else rf", $\sigma={noise_multiplier:g}$"
            )
            axis.plot(
                x_values,
                [row[n_stage_key] for row in sigma_rows],
                label=f"N-stage{suffix}",
                linewidth=2.0,
                marker="o",
            )
            axis.plot(
                x_values,
                [row[papernot_key] for row in sigma_rows],
                label=f"Papernot, breadth matched{suffix}",
                linewidth=2.0,
                linestyle="--",
                marker="x",
            )
        axis.set_xscale(x_scale)
        axis.set_xlabel(
            r"Initial conditional expected breadth $\bar\mu_1$"
        )
        axis.set_ylabel(y_label)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        plot_path = output_directory / str(output_config[filename_key])
        figure.savefig(plot_path, dpi=dpi)
        plt.close(figure)
        plot_paths[filename_key] = plot_path

    figure, axis = plt.subplots(figsize=figure_size)
    for noise_multiplier in noise_multipliers:
        sigma_rows = [
            row
            for row in rows
            if math.isclose(
                row["noise_multiplier"],
                noise_multiplier,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ]
        suffix = (
            "" if len(noise_multipliers) == 1
            else rf" ($\sigma={noise_multiplier:g}$)"
        )
        axis.plot(
            [
                row["initial_conditional_expected_runs_bar_mu_1"]
                for row in sigma_rows
            ],
            [row["epsilon_gap_n_stage_minus_papernot"] for row in sigma_rows],
            label=f"N-stage minus Papernot{suffix}",
            linewidth=2.0,
            marker="o",
        )
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    axis.set_xscale(x_scale)
    axis.set_xlabel(r"Initial conditional expected breadth $\bar\mu_1$")
    axis.set_ylabel(r"$\epsilon_{N\mathrm{-stage}}-\epsilon_P$")
    axis.set_title("Privacy Gap Relative to Breadth-Matched Papernot")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    gap_path = output_directory / str(
        output_config.privacy_gap_plot_filename
    )
    figure.savefig(gap_path, dpi=dpi)
    plt.close(figure)
    plot_paths["privacy_gap_plot_filename"] = gap_path
    return plot_paths


def epsilon_vs_mean_poisson_plot(config: DictConfig):
    """Run the fixed-schedule, breadth-matched Poisson experiment."""
    exp_config = config.experiment
    fixed_schedule_config = exp_config.fixed_schedule
    comparison_mode = str(exp_config.comparison.mode).strip().lower()
    if comparison_mode != "breadth_matched":
        raise ValueError(
            "This runner currently implements comparison.mode="
            "'breadth_matched'. Compute matching will be added as a "
            "separate matching rule."
        )

    initial_expected_counts, expected_count_scale = _build_mu_values(
        fixed_schedule_config.initial_expected_trials,
        field_name="fixed_schedule.initial_expected_trials",
    )
    num_stages = int(fixed_schedule_config.num_stages)
    retention_factor = float(fixed_schedule_config.retention_factor)
    minimum_resource = int(fixed_schedule_config.minimum_resource)
    survivor_rounding = str(fixed_schedule_config.survivor_rounding)
    resource_rounding = str(fixed_schedule_config.resource_rounding)

    delta = float(exp_config.privacy.delta)
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("privacy.delta must satisfy 0 < delta < 1.")
    max_renyi_order = int(exp_config.privacy.max_renyi_order)
    if max_renyi_order < 3:
        raise ValueError("privacy.max_renyi_order must be at least 3.")
    orders = np.arange(2, max_renyi_order + 1, dtype=float)
    noise_multipliers = tuple(
        float(value) for value in exp_config.privacy.noise_multipliers
    )
    if not noise_multipliers or any(
        not math.isfinite(value) or value <= 0.0
        for value in noise_multipliers
    ):
        raise ValueError(
            "privacy.noise_multipliers must contain finite positive values."
        )
    if len(set(noise_multipliers)) != len(noise_multipliers):
        raise ValueError("privacy.noise_multipliers must not contain duplicates.")
    sampling_rate = float(config.run_settings.sampling_rate)
    if not math.isfinite(sampling_rate) or not 0.0 < sampling_rate <= 1.0:
        raise ValueError("run_settings.sampling_rate must lie in (0, 1].")

    figure_size, dpi, x_scale = _validate_plot_settings(exp_config.plot)
    if x_scale != expected_count_scale:
        raise ValueError(
            "plot.x_scale must match fixed_schedule.initial_expected_trials."
            "scale so the CSV and figure use the same interpretation."
        )
    output_directory = Path(HydraConfig.get().runtime.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    stage_rows = []
    for noise_multiplier in noise_multipliers:
        cached_resource_signature = None
        cached_stage_base_curves = None
        cached_papernot_base_curve = None
        for initial_expected_count in initial_expected_counts:
            schedule = build_fixed_poisson_schedule(
                initial_expected_num_trials=float(initial_expected_count),
                retention_factor=retention_factor,
                num_stages=num_stages,
                minimum_resource=minimum_resource,
                survivor_rounding=survivor_rounding,
                resource_rounding=resource_rounding,
            )
            resource_signature = tuple(
                (
                    stage.cumulative_resource,
                    stage.incremental_resource,
                )
                for stage in schedule
            )
            if cached_resource_signature is None:
                (
                    cached_stage_base_curves,
                    cached_papernot_base_curve,
                ) = _compute_fixed_schedule_base_curves(
                    schedule,
                    sampling_rate=sampling_rate,
                    noise_multiplier=noise_multiplier,
                    orders=orders,
                )
                cached_resource_signature = resource_signature
            elif resource_signature != cached_resource_signature:
                raise RuntimeError(
                    "Fixed schedule resources unexpectedly changed with "
                    "the initial breadth."
                )

            result = compute_fixed_schedule_privacy(
                schedule,
                delta=delta,
                stage_base_curves=cached_stage_base_curves,
                papernot_base_curve=cached_papernot_base_curve,
            )
            initial_bar_mu = float(initial_expected_count)
            summary_rows.append(
                {
                    "initial_conditional_expected_runs_bar_mu_1": initial_bar_mu,
                    "papernot_poisson_rate_theta_p": initial_bar_mu,
                    "retention_factor_q": retention_factor,
                    "num_stages": num_stages,
                    "minimum_resource": minimum_resource,
                    "maximum_resource": schedule[-1].cumulative_resource,
                    "sampling_rate": sampling_rate,
                    "noise_multiplier": noise_multiplier,
                    "delta": delta,
                    "epsilon_n_stage": result.n_stage_dp.epsilon,
                    "epsilon_papernot_breadth_matched": (
                        result.papernot_dp.epsilon
                    ),
                    "epsilon_gap_n_stage_minus_papernot": (
                        result.n_stage_dp.epsilon
                        - result.papernot_dp.epsilon
                    ),
                    "best_order_n_stage": result.n_stage_dp.best_order,
                    "best_order_papernot": result.papernot_dp.best_order,
                    "n_stage_best_order_is_min": (
                        result.n_stage_dp.is_at_min_order
                    ),
                    "papernot_best_order_is_min": (
                        result.papernot_dp.is_at_min_order
                    ),
                    "expected_compute_n_stage": (
                        result.n_stage_expected_compute
                    ),
                    "expected_compute_papernot_breadth_matched": (
                        result.papernot_expected_compute
                    ),
                    "expected_compute_ratio_n_stage_over_papernot": (
                        result.n_stage_expected_compute
                        / result.papernot_expected_compute
                    ),
                }
            )

            total_best_index = result.n_stage_dp.best_index
            total_best_order = result.n_stage_dp.best_order
            for stage, stage_result in zip(
                schedule,
                result.n_stage.stage_results,
            ):
                is_conditioned = (
                    stage.count_semantics
                    == "conditional_mean_given_k_ge_m"
                )
                stage_rows.append(
                    {
                        "initial_conditional_expected_runs_bar_mu_1": (
                            initial_bar_mu
                        ),
                        "noise_multiplier": noise_multiplier,
                        "stage": stage.stage,
                        "is_final_stage": stage.stage == num_stages,
                        "count_semantics": stage.count_semantics,
                        "conditional_expected_runs_bar_mu_l": (
                            stage.expected_num_trials if is_conditioned else ""
                        ),
                        "unconditioned_expected_runs_theta_l": (
                            stage.expected_num_trials if not is_conditioned else ""
                        ),
                        "underlying_poisson_rate_theta_l": (
                            stage_result.distribution.mu
                        ),
                        "retained_count_m_l": stage.retained_count,
                        "planned_retention_ratio_m_l_over_bar_mu_l": (
                            stage.retained_count / stage.expected_num_trials
                        ),
                        "conditioning_probability": (
                            stage_result.distribution.survival_probability(
                                stage.retained_count
                            )
                            if is_conditioned
                            else ""
                        ),
                        "probability_k_zero": (
                            ""
                            if is_conditioned
                            else stage_result.distribution.probability_zero()
                        ),
                        "log_expected_binomial": (
                            stage_result.log_expected_binomial
                            if is_conditioned
                            else ""
                        ),
                        "cumulative_resource": stage.cumulative_resource,
                        "incremental_resource": stage.incremental_resource,
                        "stage_expected_compute": (
                            stage.expected_num_trials
                            * stage.incremental_resource
                        ),
                        "n_stage_total_best_order": total_best_order,
                        "base_rdp_at_total_best_order": (
                            stage_result.base_rdp_curve.epsilons[
                                total_best_index
                            ]
                        ),
                        "selected_stage_rdp_at_total_best_order": (
                            stage_result.rdp_curve.epsilons[
                                total_best_index
                            ]
                        ),
                        "raw_selected_stage_rdp_at_total_best_order": (
                            stage_result.raw_rdp_curve.epsilons[
                                total_best_index
                            ]
                        ),
                        "hat_epsilon_at_total_best_order": (
                            stage_result.hat_epsilons[total_best_index]
                        ),
                        "hat_delta_at_total_best_order": (
                            stage_result.hat_deltas[total_best_index]
                        ),
                        "best_auxiliary_order_at_total_best_order": (
                            stage_result.best_auxiliary_orders[
                                total_best_index
                            ]
                        ),
                    }
                )

    summary_path = output_directory / str(
        exp_config.output.summary_data_filename
    )
    summary_fieldnames = list(summary_rows[0])
    _write_dict_rows(
        summary_path,
        fieldnames=summary_fieldnames,
        rows=summary_rows,
    )
    stage_path = output_directory / str(
        exp_config.output.stage_data_filename
    )
    stage_fieldnames = list(stage_rows[0])
    _write_dict_rows(
        stage_path,
        fieldnames=stage_fieldnames,
        rows=stage_rows,
    )
    plot_paths = _plot_accounting_series(
        rows=summary_rows,
        noise_multipliers=noise_multipliers,
        figure_size=figure_size,
        dpi=dpi,
        x_scale=x_scale,
        output_directory=output_directory,
        output_config=exp_config.output,
    )

    print(f"Saved fixed-schedule summary: {summary_path}", flush=True)
    print(f"Saved per-stage diagnostics: {stage_path}", flush=True)
    for plot_path in plot_paths.values():
        print(f"Saved accounting plot: {plot_path}", flush=True)
    return {
        "summary_path": summary_path,
        "stage_path": stage_path,
        "plot_paths": plot_paths,
    }


def _read_nonempty_accounting_csv(path):
    """Read a CSV while reporting, but otherwise ignoring, blank lines."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required accounting CSV is missing: {path}")
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    blank_line_count = sum(not line.strip() for line in raw_lines)
    with path.open(mode="r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Accounting CSV has no header: {path}")
        fieldnames = [str(field).strip() for field in reader.fieldnames]
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row and any(
                str(value).strip() for value in row[None] or []
            ):
                raise ValueError(
                    f"CSV row {row_number} has extra fields in {path}."
                )
            cleaned = {
                str(key).strip(): (
                    "" if value is None else str(value).strip()
                )
                for key, value in row.items()
                if key is not None
            }
            if any(cleaned.values()):
                rows.append(cleaned)
    return fieldnames, rows, blank_line_count


def _required_float(row, field_name, *, source):
    try:
        value = float(row[field_name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{source} has an invalid {field_name!r} value."
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"{source} has non-finite {field_name!r}.")
    return value


def _required_int(row, field_name, *, source):
    value = _required_float(row, field_name, source=source)
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{source} has non-integer {field_name!r}.")
    return rounded


def _numeric_grid_key(value):
    return round(float(value), 12)


def _validate_expected_grid(actual, expected, *, name):
    actual = sorted(_numeric_grid_key(value) for value in actual)
    expected = sorted(_numeric_grid_key(value) for value in expected)
    if actual != expected:
        raise ValueError(
            f"The compiled {name} grid does not match the configured grid. "
            f"Actual={actual}; expected={expected}."
        )


def _q_mu_matrix(rows, *, q_values, mu_values, field_name):
    lookup = {
        (
            _numeric_grid_key(row["retention_factor_q"]),
            _numeric_grid_key(
                row["initial_conditional_expected_runs_bar_mu_1"]
            ),
        ): float(row[field_name])
        for row in rows
    }
    matrix = np.empty((len(q_values), len(mu_values)), dtype=float)
    for q_index, q_value in enumerate(q_values):
        for mu_index, mu_value in enumerate(mu_values):
            key = (
                _numeric_grid_key(q_value),
                _numeric_grid_key(mu_value),
            )
            try:
                matrix[q_index, mu_index] = lookup[key]
            except KeyError as error:
                raise ValueError(
                    f"Missing {field_name} at q={q_value}, "
                    f"bar_mu_1={mu_value}."
                ) from error
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"The {field_name} heatmap contains non-finite data.")
    return matrix


def _set_q_mu_heatmap_axes(axis, *, q_values, mu_values):
    x_stride = max(1, int(math.ceil(len(mu_values) / 10.0)))
    x_indices = list(range(0, len(mu_values), x_stride))
    if x_indices[-1] != len(mu_values) - 1:
        x_indices.append(len(mu_values) - 1)
    axis.set_xticks(x_indices)
    axis.set_xticklabels(
        [f"{mu_values[index]:g}" for index in x_indices],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(np.arange(len(q_values)))
    axis.set_yticklabels([f"{value:g}" for value in q_values])
    axis.set_xlabel(r"Initial conditional expected breadth $\bar\mu_1$")
    axis.set_ylabel(r"Retention parameter $q$")


def _save_scalar_q_mu_heatmap(
    *,
    rows,
    q_values,
    mu_values,
    field_name,
    title,
    colorbar_label,
    output_path,
    figure_size,
    dpi,
    cmap="viridis",
    norm=None,
):
    matrix = _q_mu_matrix(
        rows,
        q_values=q_values,
        mu_values=mu_values,
        field_name=field_name,
    )
    figure, axis = plt.subplots(figsize=figure_size)
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    _set_q_mu_heatmap_axes(
        axis,
        q_values=q_values,
        mu_values=mu_values,
    )
    axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _save_stage_panel_heatmaps(
    *,
    rows,
    q_values,
    mu_values,
    field_names,
    titles,
    output_path,
    dpi,
    cmap,
    colorbar_label,
    norm=None,
    annotate_integers=False,
):
    if len(field_names) != len(titles) or not field_names:
        raise ValueError("Stage heatmap fields and titles must be non-empty.")
    figure, axes = plt.subplots(
        1,
        len(field_names),
        figsize=(5.0 * len(field_names), 5.2),
        squeeze=False,
        sharey=True,
    )
    axes = axes[0]
    image = None
    for axis, field_name, title in zip(axes, field_names, titles):
        matrix = _q_mu_matrix(
            rows,
            q_values=q_values,
            mu_values=mu_values,
            field_name=field_name,
        )
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
        )
        _set_q_mu_heatmap_axes(
            axis,
            q_values=q_values,
            mu_values=mu_values,
        )
        axis.set_title(title)
        if annotate_integers:
            threshold = (
                float(np.nanmin(matrix)) + float(np.nanmax(matrix))
            ) / 2.0
            for q_index in range(matrix.shape[0]):
                for mu_index in range(matrix.shape[1]):
                    value = matrix[q_index, mu_index]
                    axis.text(
                        mu_index,
                        q_index,
                        f"{int(round(value))}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color=("white" if value > threshold else "black"),
                    )
    figure.subplots_adjust(
        left=0.07,
        right=0.88,
        bottom=0.20,
        top=0.88,
        wspace=0.24,
    )
    colorbar_axis = figure.add_axes([0.91, 0.20, 0.018, 0.65])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(colorbar_label)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def compile_fixed_schedule_q_sweep(config: DictConfig):
    """Validate separate fixed-schedule runs and compile cross-q maps."""
    exp_config = config.experiment
    sweep_config = exp_config.q_sweep
    input_root = Path(str(sweep_config.input_root))
    source_directories = sorted(
        path
        for path in input_root.glob(str(sweep_config.source_glob))
        if path.is_dir()
    )
    if not source_directories:
        raise FileNotFoundError(
            f"No q-sweep directories matched beneath {input_root}."
        )

    expected_q_values = sorted(
        float(value) for value in sweep_config.expected_q_values
    )
    if not expected_q_values or any(
        not math.isfinite(value) or not 0.0 < value < 1.0
        for value in expected_q_values
    ):
        raise ValueError(
            "q_sweep.expected_q_values must contain values in (0, 1)."
        )
    target_noise_multiplier = float(
        sweep_config.target_noise_multiplier
    )
    summary_filename = str(sweep_config.summary_filename)
    stage_filename = str(sweep_config.stage_filename)

    summary_by_key = {}
    stage_by_key = {}
    validation_rows = []
    summary_fieldnames = None
    stage_fieldnames = None
    source_q_values = []
    for source_directory in source_directories:
        (
            current_summary_fieldnames,
            summary_rows,
            summary_blank_lines,
        ) = _read_nonempty_accounting_csv(
            source_directory / summary_filename
        )
        (
            current_stage_fieldnames,
            stage_rows,
            stage_blank_lines,
        ) = _read_nonempty_accounting_csv(source_directory / stage_filename)
        if summary_fieldnames is None:
            summary_fieldnames = current_summary_fieldnames
            stage_fieldnames = current_stage_fieldnames
        elif current_summary_fieldnames != summary_fieldnames:
            raise ValueError(
                f"Summary schema mismatch in {source_directory}."
            )
        elif current_stage_fieldnames != stage_fieldnames:
            raise ValueError(f"Stage schema mismatch in {source_directory}.")

        q_values_in_source = {
            _numeric_grid_key(
                _required_float(
                    row,
                    "retention_factor_q",
                    source=source_directory,
                )
            )
            for row in summary_rows
        }
        if len(q_values_in_source) != 1:
            raise ValueError(
                f"{source_directory} must contain exactly one q value."
            )
        source_q = float(next(iter(q_values_in_source)))
        source_q_values.append(source_q)

        summary_duplicate_count = 0
        for row in summary_rows:
            sigma = _required_float(
                row,
                "noise_multiplier",
                source=source_directory,
            )
            if not math.isclose(
                sigma,
                target_noise_multiplier,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{source_directory} uses sigma={sigma}, expected "
                    f"{target_noise_multiplier}."
                )
            mu = _required_float(
                row,
                "initial_conditional_expected_runs_bar_mu_1",
                source=source_directory,
            )
            key = (
                _numeric_grid_key(source_q),
                _numeric_grid_key(mu),
            )
            previous = summary_by_key.get(key)
            if previous is not None:
                if previous["row"] != row:
                    raise ValueError(
                        f"Conflicting summary rows for q={source_q}, "
                        f"bar_mu_1={mu}."
                    )
                summary_duplicate_count += 1
            else:
                summary_by_key[key] = {
                    "row": row,
                    "source": source_directory.name,
                }

        stage_duplicate_count = 0
        for row in stage_rows:
            sigma = _required_float(
                row,
                "noise_multiplier",
                source=source_directory,
            )
            if not math.isclose(
                sigma,
                target_noise_multiplier,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{source_directory} has an unexpected stage sigma."
                )
            mu = _required_float(
                row,
                "initial_conditional_expected_runs_bar_mu_1",
                source=source_directory,
            )
            stage = _required_int(
                row,
                "stage",
                source=source_directory,
            )
            key = (
                _numeric_grid_key(source_q),
                _numeric_grid_key(mu),
                stage,
            )
            previous = stage_by_key.get(key)
            if previous is not None:
                if previous["row"] != row:
                    raise ValueError(
                        f"Conflicting stage rows for q={source_q}, "
                        f"bar_mu_1={mu}, stage={stage}."
                    )
                stage_duplicate_count += 1
            else:
                stage_by_key[key] = {
                    "row": row,
                    "source": source_directory.name,
                }

        validation_rows.append(
            {
                "source_run_directory": source_directory.name,
                "retention_factor_q": source_q,
                "summary_nonempty_rows": len(summary_rows),
                "summary_identical_duplicates_removed": (
                    summary_duplicate_count
                ),
                "summary_blank_lines_ignored": summary_blank_lines,
                "stage_nonempty_rows": len(stage_rows),
                "stage_identical_duplicates_removed": stage_duplicate_count,
                "stage_blank_lines_ignored": stage_blank_lines,
            }
        )

    _validate_expected_grid(
        source_q_values,
        expected_q_values,
        name="q",
    )
    q_values = expected_q_values
    mu_values_by_q = {}
    for q_value, mu_value in summary_by_key:
        mu_values_by_q.setdefault(q_value, []).append(mu_value)
    reference_mu_values = None
    for q_value in q_values:
        mu_values = sorted(
            set(mu_values_by_q.get(_numeric_grid_key(q_value), []))
        )
        if reference_mu_values is None:
            reference_mu_values = mu_values
        elif mu_values != reference_mu_values:
            raise ValueError(
                f"The bar_mu_1 grid for q={q_value} is incomplete or "
                "inconsistent."
            )
    if not reference_mu_values:
        raise ValueError("The q sweep contains no bar_mu_1 values.")
    mu_values = [float(value) for value in reference_mu_values]

    constant_fields = (
        "num_stages",
        "minimum_resource",
        "sampling_rate",
        "noise_multiplier",
        "delta",
    )
    for field_name in constant_fields:
        values = {
            _numeric_grid_key(
                _required_float(
                    item["row"],
                    field_name,
                    source=item["source"],
                )
            )
            for item in summary_by_key.values()
        }
        if len(values) != 1:
            raise ValueError(
                f"The q sweep changes supposedly fixed field {field_name}: "
                f"{sorted(values)}."
            )
    num_stages = _required_int(
        next(iter(summary_by_key.values()))["row"],
        "num_stages",
        source="q sweep",
    )

    combined_summary_rows = []
    combined_stage_rows = []
    for q_value in q_values:
        for mu_value in mu_values:
            point_key = (
                _numeric_grid_key(q_value),
                _numeric_grid_key(mu_value),
            )
            try:
                summary_item = summary_by_key[point_key]
            except KeyError as error:
                raise ValueError(
                    f"Missing summary point q={q_value}, "
                    f"bar_mu_1={mu_value}."
                ) from error
            point_stage_items = []
            for stage in range(1, num_stages + 1):
                stage_key = (*point_key, stage)
                try:
                    point_stage_items.append(stage_by_key[stage_key])
                except KeyError as error:
                    raise ValueError(
                        f"Missing stage point q={q_value}, "
                        f"bar_mu_1={mu_value}, stage={stage}."
                    ) from error

            rdp_contributions = [
                _required_float(
                    item["row"],
                    "selected_stage_rdp_at_total_best_order",
                    source=item["source"],
                )
                for item in point_stage_items
            ]
            total_rdp_contribution = sum(rdp_contributions)
            if total_rdp_contribution <= 0.0:
                raise ValueError(
                    f"Non-positive total stage RDP at q={q_value}, "
                    f"bar_mu_1={mu_value}."
                )
            final_stage_row = point_stage_items[-1]["row"]
            probability_k_zero = _required_float(
                final_stage_row,
                "probability_k_zero",
                source=point_stage_items[-1]["source"],
            )
            if not 0.0 <= probability_k_zero <= 1.0:
                raise ValueError("Final-stage K=0 probability is invalid.")

            combined_summary_row = dict(summary_item["row"])
            combined_summary_row["source_run_directory"] = summary_item[
                "source"
            ]
            combined_summary_row[
                "final_stage_probability_k_zero"
            ] = probability_k_zero
            for stage, (stage_item, contribution) in enumerate(
                zip(point_stage_items, rdp_contributions),
                start=1,
            ):
                combined_summary_row[f"retained_count_m_{stage}"] = (
                    _required_int(
                        stage_item["row"],
                        "retained_count_m_l",
                        source=stage_item["source"],
                    )
                )
                combined_summary_row[f"stage_{stage}_rdp_share"] = (
                    contribution / total_rdp_contribution
                )
                combined_stage_row = {
                    "retention_factor_q": q_value,
                    "source_run_directory": stage_item["source"],
                    **stage_item["row"],
                    "rdp_share_at_total_best_order": (
                        contribution / total_rdp_contribution
                    ),
                }
                combined_stage_rows.append(combined_stage_row)
            combined_summary_rows.append(combined_summary_row)

    output_directory = Path(HydraConfig.get().runtime.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_config = exp_config.output
    combined_summary_path = output_directory / str(
        output_config.combined_summary_filename
    )
    _write_dict_rows(
        combined_summary_path,
        fieldnames=list(combined_summary_rows[0]),
        rows=combined_summary_rows,
    )
    combined_stage_path = output_directory / str(
        output_config.combined_stage_filename
    )
    _write_dict_rows(
        combined_stage_path,
        fieldnames=list(combined_stage_rows[0]),
        rows=combined_stage_rows,
    )
    validation_path = output_directory / str(
        output_config.input_validation_filename
    )
    _write_dict_rows(
        validation_path,
        fieldnames=list(validation_rows[0]),
        rows=validation_rows,
    )

    figure_size = tuple(float(value) for value in exp_config.plot.figsize)
    if len(figure_size) != 2 or any(value <= 0.0 for value in figure_size):
        raise ValueError("plot.figsize must contain two positive values.")
    dpi = int(exp_config.plot.dpi)
    if dpi <= 0:
        raise ValueError("plot.dpi must be positive.")

    epsilon_path = output_directory / str(
        output_config.epsilon_heatmap_filename
    )
    _save_scalar_q_mu_heatmap(
        rows=combined_summary_rows,
        q_values=q_values,
        mu_values=mu_values,
        field_name="epsilon_n_stage",
        title="N-Stage Privacy Budget",
        colorbar_label=r"$\epsilon_{N\mathrm{-stage}}$",
        output_path=epsilon_path,
        figure_size=figure_size,
        dpi=dpi,
    )

    gap_matrix = _q_mu_matrix(
        combined_summary_rows,
        q_values=q_values,
        mu_values=mu_values,
        field_name="epsilon_gap_n_stage_minus_papernot",
    )
    gap_min = float(np.min(gap_matrix))
    gap_max = float(np.max(gap_matrix))
    gap_norm = (
        TwoSlopeNorm(vmin=gap_min, vcenter=0.0, vmax=gap_max)
        if gap_min < 0.0 < gap_max
        else None
    )
    gap_path = output_directory / str(
        output_config.privacy_gap_heatmap_filename
    )
    _save_scalar_q_mu_heatmap(
        rows=combined_summary_rows,
        q_values=q_values,
        mu_values=mu_values,
        field_name="epsilon_gap_n_stage_minus_papernot",
        title="Privacy Gap: N-Stage Minus Breadth-Matched Papernot",
        colorbar_label=r"$\epsilon_{N\mathrm{-stage}}-\epsilon_P$",
        output_path=gap_path,
        figure_size=figure_size,
        dpi=dpi,
        cmap="coolwarm",
        norm=gap_norm,
    )

    compute_ratio_path = output_directory / str(
        output_config.expected_compute_ratio_heatmap_filename
    )
    _save_scalar_q_mu_heatmap(
        rows=combined_summary_rows,
        q_values=q_values,
        mu_values=mu_values,
        field_name="expected_compute_ratio_n_stage_over_papernot",
        title="Expected Compute Ratio",
        colorbar_label=r"$E[C_N]/E[C_P]$",
        output_path=compute_ratio_path,
        figure_size=figure_size,
        dpi=dpi,
    )

    probability_matrix = _q_mu_matrix(
        combined_summary_rows,
        q_values=q_values,
        mu_values=mu_values,
        field_name="final_stage_probability_k_zero",
    )
    positive_probabilities = probability_matrix[probability_matrix > 0.0]
    if positive_probabilities.size == 0:
        raise ValueError("Every final-stage K=0 probability is zero.")
    probability_path = output_directory / str(
        output_config.final_stage_zero_probability_heatmap_filename
    )
    _save_scalar_q_mu_heatmap(
        rows=combined_summary_rows,
        q_values=q_values,
        mu_values=mu_values,
        field_name="final_stage_probability_k_zero",
        title="Probability the Final Poisson Stage Is Empty",
        colorbar_label=r"$P(K_L=0)$ (log scale)",
        output_path=probability_path,
        figure_size=figure_size,
        dpi=dpi,
        cmap="magma",
        norm=LogNorm(
            vmin=float(np.min(positive_probabilities)),
            vmax=float(np.max(positive_probabilities)),
        ),
    )

    rdp_share_path = output_directory / str(
        output_config.stage_rdp_share_heatmaps_filename
    )
    _save_stage_panel_heatmaps(
        rows=combined_summary_rows,
        q_values=q_values,
        mu_values=mu_values,
        field_names=[
            f"stage_{stage}_rdp_share"
            for stage in range(1, num_stages + 1)
        ],
        titles=[
            f"Stage {stage} RDP share"
            for stage in range(1, num_stages + 1)
        ],
        output_path=rdp_share_path,
        dpi=dpi,
        cmap="viridis",
        colorbar_label="Fraction of composed RDP at selected order",
        norm=Normalize(vmin=0.0, vmax=1.0),
    )

    retained_count_path = output_directory / str(
        output_config.retained_count_heatmaps_filename
    )
    _save_stage_panel_heatmaps(
        rows=combined_summary_rows,
        q_values=q_values,
        mu_values=mu_values,
        field_names=[
            f"retained_count_m_{stage}"
            for stage in range(1, num_stages)
        ],
        titles=[
            rf"Stage {stage} retained count $m_{stage}$"
            for stage in range(1, num_stages)
        ],
        output_path=retained_count_path,
        dpi=dpi,
        cmap="YlGnBu",
        colorbar_label="Retained run occurrences",
        annotate_integers=True,
    )

    plot_paths = {
        "epsilon": epsilon_path,
        "privacy_gap": gap_path,
        "expected_compute_ratio": compute_ratio_path,
        "final_stage_zero_probability": probability_path,
        "stage_rdp_shares": rdp_share_path,
        "retained_counts": retained_count_path,
    }
    print(f"Saved combined q-sweep summary: {combined_summary_path}", flush=True)
    print(f"Saved combined q-sweep stages: {combined_stage_path}", flush=True)
    print(f"Saved q-sweep input validation: {validation_path}", flush=True)
    for plot_path in plot_paths.values():
        print(f"Saved q-sweep heatmap: {plot_path}", flush=True)
    return {
        "combined_summary_path": combined_summary_path,
        "combined_stage_path": combined_stage_path,
        "validation_path": validation_path,
        "plot_paths": plot_paths,
    }


EXPERIMENT_RUNNERS = {
    "compile_fixed_schedule_q_sweep": compile_fixed_schedule_q_sweep,
    "coverage_probability_vs_mean": coverage_probability_vs_mean_plot,
    "epsilon_vs_mean_poisson": epsilon_vs_mean_poisson_plot,
}


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config_cl",
)
def main(config: DictConfig) -> None:
    runner_name = str(config.experiment.get("runner", ""))
    try:
        runner = EXPERIMENT_RUNNERS[runner_name]
    except KeyError as error:
        available_runners = ", ".join(sorted(EXPERIMENT_RUNNERS))
        raise ValueError(
            f"Unknown privacy-accounting runner {runner_name!r}. "
            f"Available runners: {available_runners}."
        ) from error
    runner(config)


if __name__ == "__main__":
    main()
