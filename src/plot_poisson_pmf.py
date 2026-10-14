"""Plot the PMF implemented by ``PoissonDistribution``.

Example:
    python src/plot_poisson_pmf.py --mu 10
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from privacy_accounting.poisson import PoissonDistribution


DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "scratch_pad_plots"
)


def plot_poisson_pmf(
    *,
    mu: float,
    max_k: int | None = None,
    output_path: Path | None = None,
) -> Path:
    """Plot and save the PMF of ``Poisson(mu)``.

    When ``max_k`` is omitted, the plot includes at least four standard
    deviations above the mean and always shows integer values 0 through 10.
    """
    distribution = PoissonDistribution(mu=mu)

    if max_k is None:
        max_k = max(10, math.ceil(mu + 4.0 * math.sqrt(mu)))
    if isinstance(max_k, bool) or not isinstance(max_k, int) or max_k < 0:
        raise ValueError("max_k must be a non-negative integer.")

    k_values = np.arange(max_k + 1, dtype=int)
    probabilities = np.asarray(
        [distribution.pmf(int(k)) for k in k_values],
        dtype=float,
    )
    if not np.all(np.isfinite(probabilities)):
        raise ArithmeticError("The computed PMF contains a non-finite value.")

    if output_path is None:
        output_path = (
            DEFAULT_OUTPUT_DIRECTORY / f"poisson_pmf_mu_{mu:g}.png"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    markerline, stemlines, baseline = axis.stem(
        k_values,
        probabilities,
        basefmt=" ",
    )
    plt.setp(markerline, markersize=5, color="tab:blue")
    plt.setp(stemlines, linewidth=1.5, color="tab:blue")
    baseline.set_visible(False)

    axis.set_xlabel("Number of events, k")
    axis.set_ylabel("Probability, P(K = k)")
    axis.set_title(rf"Poisson PMF ($\mu={mu:g}$)")
    axis.set_xticks(k_values)
    axis.grid(axis="y", alpha=0.3)
    axis.set_ylim(bottom=0.0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a Poisson PMF using PoissonDistribution.pmf()."
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=10.0,
        help="Underlying Poisson rate (default: 10).",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="Largest integer k to display (default: automatic).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path (default: "
            "scratch_pad_plots/poisson_pmf_mu_<mu>.png)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = plot_poisson_pmf(
        mu=args.mu,
        max_k=args.max_k,
        output_path=args.output,
    )
    print(f"Saved Poisson PMF plot to {output_path}")


if __name__ == "__main__":
    main()
