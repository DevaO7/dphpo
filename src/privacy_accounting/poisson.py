"""Poisson trial-count utilities for private HPO selection mechanisms.

``PoissonDistribution`` represents the unconditioned variable

    K0 ~ Poisson(mu),

supported on ``{0, 1, ...}``.  Direct Papernot--Steinke top-1 selection
uses this variable without conditioning and defines a public fallback for
``K0 = 0``.  Conditioned top-m selection instead uses

    K = K0 | K0 >= m.

The class exposes both mechanisms without silently changing the meaning of
``mu``: ``mu`` is always the underlying Poisson rate, while
``conditional_mean(m)`` is the expected number of runs after conditioning.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincc

from .validation import validate_positive_integer


_MU_LOWER = 1e-12


def _validate_mu(mu: float) -> float:
    mu = float(mu)
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and strictly positive.")
    return mu


def _survival_probability(mu: float, m: int) -> float:
    """Return ``P[Poisson(mu) >= m]`` without subtractive cancellation."""
    mu = _validate_mu(mu)
    m = validate_positive_integer(m, "m")
    survival = float(gammainc(m, mu))
    if not math.isfinite(survival) or survival <= 0.0:
        raise ArithmeticError(
            "P[Poisson(mu) >= m] is numerically non-positive."
        )
    return min(survival, 1.0)


def _conditional_mean(mu: float, m: int) -> float:
    """Return ``E[K0 | K0 >= m]`` for ``K0 ~ Poisson(mu)``."""
    mu = _validate_mu(mu)
    m = validate_positive_integer(m, "m")
    numerator_survival = (
        1.0
        if m == 1
        else _survival_probability(mu, m - 1)
    )
    return (
        mu
        * numerator_survival
        / _survival_probability(mu, m)
    )


def _solve_mu_for_conditional_mean(
    *,
    m: int,
    target_mean: float,
) -> float:
    """Solve ``E[K0 | K0 >= m] = target_mean`` for the Poisson rate."""
    m = validate_positive_integer(m, "m")
    target_mean = float(target_mean)
    if not math.isfinite(target_mean):
        raise ValueError("target_mean must be finite.")
    if target_mean <= m:
        raise ValueError(
            f"target_mean must be strictly greater than m={m}. For every "
            "finite mu > 0, Poisson(mu) conditioned on K0 >= m has "
            "positive probability above m; equality is attained only in "
            "the limiting degenerate mechanism mu -> 0."
        )

    def objective(mu: float) -> float:
        return _conditional_mean(mu, m) - target_mean

    lower = min(_MU_LOWER, target_mean / 2.0)
    lower_value = objective(lower)
    upper = target_mean
    upper_value = objective(upper)

    if lower_value >= 0.0:
        raise ValueError(
            "Could not bracket mu below the requested conditional mean. "
            "The target may be too close to m for the supported numerical "
            "precision."
        )
    if upper_value < 0.0:
        raise ArithmeticError(
            "Could not bracket the conditioned-Poisson rate even though "
            "the conditional mean should exceed the underlying rate."
        )
    if upper_value == 0.0:
        return float(upper)

    return float(
        brentq(
            objective,
            lower,
            upper,
            xtol=1e-12,
            rtol=1e-12,
            maxiter=300,
        )
    )


@dataclass(frozen=True, slots=True)
class PoissonDistribution:
    """A Poisson random variable supported on ``{0, 1, ...}``."""

    mu: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mu", _validate_mu(self.mu))

    @classmethod
    def from_mean(cls, *, target_mean: float) -> "PoissonDistribution":
        """Construct ``K0`` satisfying ``E[K0] = target_mean``."""
        return cls(mu=target_mean)

    @classmethod
    def from_conditional_mean(
        cls,
        *,
        m: int,
        target_mean: float,
    ) -> "PoissonDistribution":
        """Construct ``K0`` satisfying ``E[K0 | K0 >= m] = target_mean``."""
        return cls(
            mu=_solve_mu_for_conditional_mean(
                m=m,
                target_mean=target_mean,
            )
        )

    def mean(self) -> float:
        """Return ``E[K]``."""
        return self.mu

    def pmf(self, k: int) -> float:
        """Return ``P[K = k]`` using a stable log-domain formula."""
        if not isinstance(k, Integral):
            raise TypeError("k must be an integer.")
        k = int(k)
        if k < 0:
            return 0.0
        log_probability = (
            -self.mu
            + k * math.log(self.mu)
            - math.lgamma(k + 1)
        )
        return min(max(math.exp(log_probability), 0.0), 1.0)

    def probability_zero(self) -> float:
        """Return ``P[K0 = 0]``."""
        return math.exp(-self.mu)

    def probability_less_than(self, m: int) -> float:
        """Return ``P[K0 < m]``."""
        m = validate_positive_integer(m, "m")
        probability = float(gammaincc(m, self.mu))
        if not math.isfinite(probability):
            raise ArithmeticError(
                "P[Poisson(mu) < m] is not finite."
            )
        return min(max(probability, 0.0), 1.0)

    def survival_probability(self, m: int) -> float:
        """Return ``P[K0 >= m]``."""
        return _survival_probability(self.mu, m)

    def conditional_mean(self, m: int) -> float:
        """Return ``E[K0 | K0 >= m]``."""
        return _conditional_mean(self.mu, m)

    def sample(
        self,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Draw one trial count, including zero when it occurs."""
        if rng is None:
            rng = np.random.default_rng()
        return int(rng.poisson(self.mu))

    def sample_conditional(
        self,
        m: int,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Draw ``K0`` conditioned on ``K0 >= m``."""
        m = validate_positive_integer(m, "m")
        if rng is None:
            rng = np.random.default_rng()

        log_probability = (
            -self.mu
            + m * math.log(self.mu)
            - math.lgamma(m + 1)
            - math.log(self.survival_probability(m))
        )
        probability = min(math.exp(log_probability), 1.0)
        cumulative_probability = probability
        uniform = float(rng.random())
        k = m

        while uniform >= cumulative_probability:
            probability *= self.mu / (k + 1.0)
            k += 1
            next_cumulative_probability = (
                cumulative_probability + probability
            )
            if next_cumulative_probability == cumulative_probability:
                raise ArithmeticError(
                    "The conditioned-Poisson tail became too small to "
                    "sample numerically."
                )
            cumulative_probability = next_cumulative_probability

        return k

    def log_expected_binomial(self, m: int) -> float:
        r"""Return ``log E[binom(K, m)]`` for ``K = K0 | K0 >= m``."""
        m = validate_positive_integer(m, "m")
        return (
            m * math.log(self.mu)
            - math.lgamma(m + 1)
            - math.log(self.survival_probability(m))
        )
