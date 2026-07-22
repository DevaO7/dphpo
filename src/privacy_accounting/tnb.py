"""
Truncated negative binomial utilities for Papernot-Steinke private HPO.

This module represents the base random variable K0 supported on {1, 2, ...}.
For top-m accounting, the relevant conditioned variable is

    K = K0 | K0 >= m.

The public interface is TNBDistribution. Gamma-solving is implemented through
private module-level functions and exposed through alternative class
constructors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, gammasgn


_ETA_ZERO_TOL = 1e-12
_GAMMA_LOWER = 1e-12
_GAMMA_UPPER = 1.0 - 1e-12


def _validate_eta(eta: float) -> float:
    eta = float(eta)

    if not math.isfinite(eta):
        raise ValueError("eta must be finite.")
    if eta <= -1.0:
        raise ValueError("eta must satisfy eta > -1.")

    return eta


def _validate_gamma(gamma: float) -> float:
    gamma = float(gamma)

    if not math.isfinite(gamma):
        raise ValueError("gamma must be finite.")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must satisfy 0 < gamma < 1.")

    return gamma


def _validate_positive_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if value < 1:
        raise ValueError(f"{name} must be at least 1.")

    return value


def _is_eta_zero(eta: float) -> bool:
    return abs(eta) < _ETA_ZERO_TOL


def _mean_from_parameters(eta: float, gamma: float) -> float:
    """
    Compute the unconditional mean E[K0].
    """
    eta = _validate_eta(eta)
    gamma = _validate_gamma(gamma)

    if _is_eta_zero(eta):
        # Equivalent to:
        # (1 / gamma - 1) / log(1 / gamma)
        return (1.0 - gamma) / (gamma * (-math.log(gamma)))

    # Stable computation of 1 - gamma**eta.
    denominator_factor = -math.expm1(eta * math.log(gamma))

    return eta * (1.0 - gamma) / (gamma * denominator_factor)


def _pmf_from_parameters(k: int, eta: float, gamma: float) -> float:
    """
    Compute P(K0 = k), where K0 is supported on {1, 2, ...}.
    """
    eta = _validate_eta(eta)
    gamma = _validate_gamma(gamma)

    if not isinstance(k, Integral):
        raise TypeError("k must be an integer.")

    k = int(k)

    if k < 1:
        return 0.0

    log_one_minus_gamma = math.log1p(-gamma)

    if _is_eta_zero(eta):
        log_probability = (
            k * log_one_minus_gamma
            - math.log(k)
            - math.log(-math.log(gamma))
        )
        return math.exp(log_probability)

    # (eta)_k = Gamma(eta + k) / Gamma(eta).
    # For eta in (-1, 0), both the rising factorial and the normalization
    # denominator are negative, so their ratio remains positive.
    log_abs_rising_factorial = gammaln(eta + k) - gammaln(eta)
    sign_rising_factorial = gammasgn(eta + k) * gammasgn(eta)

    normalization = math.expm1(-eta * math.log(gamma))
    sign_normalization = 1.0 if normalization > 0.0 else -1.0

    if sign_rising_factorial * sign_normalization <= 0.0:
        raise ArithmeticError(
            "Unexpected sign while evaluating the TNB probability mass."
        )

    log_probability = (
        log_abs_rising_factorial
        - math.lgamma(k + 1)
        + k * log_one_minus_gamma
        - math.log(abs(normalization))
    )

    probability = math.exp(log_probability)

    # Protect against tiny floating-point overshoots.
    return min(max(probability, 0.0), 1.0)


def _probability_less_than_from_parameters(
    m: int,
    eta: float,
    gamma: float,
) -> float:
    """
    Compute P(K0 < m).
    """
    m = _validate_positive_integer(m, "m")

    probability = math.fsum(
        _pmf_from_parameters(k, eta, gamma)
        for k in range(1, m)
    )

    return min(max(probability, 0.0), 1.0)


def _conditional_mean_from_parameters(
    m: int,
    eta: float,
    gamma: float,
) -> float:
    """
    Compute E[K0 | K0 >= m].
    """
    m = _validate_positive_integer(m, "m")

    probability_less_than_m = _probability_less_than_from_parameters(
        m, eta, gamma
    )
    survival_probability = 1.0 - probability_less_than_m

    if survival_probability <= 0.0:
        raise ArithmeticError(
            "P(K0 >= m) is numerically non-positive."
        )

    lower_first_moment = math.fsum(
        k * _pmf_from_parameters(k, eta, gamma)
        for k in range(1, m)
    )

    tail_first_moment = (
        _mean_from_parameters(eta, gamma) - lower_first_moment
    )

    if tail_first_moment < 0.0 and abs(tail_first_moment) < 1e-14:
        tail_first_moment = 0.0

    if tail_first_moment < 0.0:
        raise ArithmeticError(
            "The tail first moment became negative numerically."
        )

    return tail_first_moment / survival_probability


def _solve_gamma_for_mean(
    eta: float,
    target_mean: float,
) -> float:
    """
    Solve E[K0] = target_mean for gamma.
    """
    eta = _validate_eta(eta)
    target_mean = float(target_mean)

    if not math.isfinite(target_mean):
        raise ValueError("target_mean must be finite.")
    if target_mean <= 1.0:
        raise ValueError(
            "target_mean must be greater than 1 because K0 >= 1."
        )

    def objective(gamma: float) -> float:
        return _mean_from_parameters(eta, gamma) - target_mean

    lower_value = objective(_GAMMA_LOWER)
    upper_value = objective(_GAMMA_UPPER)

    if lower_value * upper_value > 0.0:
        raise ValueError(
            "Could not bracket gamma for the requested mean. "
            "The requested target may be outside the numerically supported "
            "range."
        )

    return float(
        brentq(
            objective,
            _GAMMA_LOWER,
            _GAMMA_UPPER,
            xtol=1e-12,
            rtol=1e-12,
            maxiter=300,
        )
    )


def _solve_gamma_for_conditional_mean(
    eta: float,
    m: int,
    target_mean: float,
) -> float:
    """
    Solve E[K0 | K0 >= m] = target_mean for gamma.
    """
    eta = _validate_eta(eta)
    m = _validate_positive_integer(m, "m")
    target_mean = float(target_mean)

    if not math.isfinite(target_mean):
        raise ValueError("target_mean must be finite.")

    if target_mean <= m:
        raise ValueError(
            f"target_mean must be greater than m={m}, because "
            "K0 | K0 >= m is supported on {m, m+1, ...}."
        )

    def objective(gamma: float) -> float:
        return (
            _conditional_mean_from_parameters(
                m=m,
                eta=eta,
                gamma=gamma,
            )
            - target_mean
        )

    lower = 1e-12
    lower_value = objective(lower)

    # Do not immediately evaluate at gamma extremely close to 1.
    # The survival probability can suffer catastrophic cancellation there.
    upper_candidates = (
        0.5,
        0.8,
        0.9,
        0.95,
        0.99,
        0.999,
        0.9999,
        0.99999,
        0.999999,
        1.0 - 1e-8,
        1.0 - 1e-10,
    )

    if lower_value == 0.0:
        return float(lower)

    for upper in upper_candidates:
        try:
            upper_value = objective(upper)
        except ArithmeticError:
            # Survival probability was too small to evaluate reliably.
            continue

        if lower_value * upper_value <= 0.0:
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

    raise ValueError(
        "Could not bracket gamma for the requested conditional mean. "
        f"target_mean={target_mean}, m={m}, eta={eta}."
    )

@dataclass(frozen=True)
class TNBDistribution:
    """
    Papernot-Steinke truncated negative binomial distribution.

    Parameters
    ----------
    eta:
        Shape parameter satisfying eta > -1.

    gamma:
        Distribution parameter satisfying 0 < gamma < 1.
    """

    eta: float
    gamma: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "eta", _validate_eta(self.eta))
        object.__setattr__(self, "gamma", _validate_gamma(self.gamma))

    @classmethod
    def from_mean(
        cls,
        *,
        eta: float,
        target_mean: float,
    ) -> "TNBDistribution":
        """
        Construct a TNB distribution satisfying E[K0] = target_mean.
        """
        gamma = _solve_gamma_for_mean(
            eta=eta,
            target_mean=target_mean,
        )
        return cls(eta=eta, gamma=gamma)

    @classmethod
    def from_conditional_mean(
        cls,
        *,
        eta: float,
        m: int,
        target_mean: float,
    ) -> "TNBDistribution":
        """
        Construct a TNB distribution satisfying

            E[K0 | K0 >= m] = target_mean.
        """
        gamma = _solve_gamma_for_conditional_mean(
            eta=eta,
            m=m,
            target_mean=target_mean,
        )
        return cls(eta=eta, gamma=gamma)

    def mean(self) -> float:
        """
        Return the unconditional mean E[K0].
        """
        return _mean_from_parameters(self.eta, self.gamma)

    def pmf(self, k: int) -> float:
        """
        Return P(K0 = k).
        """
        return _pmf_from_parameters(k, self.eta, self.gamma)

    def probability_less_than(self, m: int) -> float:
        """
        Return P(K0 < m).
        """
        return _probability_less_than_from_parameters(
            m, self.eta, self.gamma
        )

    def survival_probability(self, m: int) -> float:
        """
        Return P(K0 >= m).
        """
        survival = 1.0 - self.probability_less_than(m)

        if survival <= 0.0:
            raise ArithmeticError(
                "P(K0 >= m) is numerically non-positive."
            )

        return survival

    def conditional_mean(self, m: int) -> float:
        """
        Return E[K0 | K0 >= m].
        """
        return _conditional_mean_from_parameters(
            m, self.eta, self.gamma
        )

    def sample(self, rng: np.random.Generator | None = None) -> int:
        """Draw and return one value of K0 from this distribution."""
        if rng is None:
            rng = np.random.default_rng()

        one_minus_gamma = 1.0 - self.gamma

        if _is_eta_zero(self.eta):
            return int(rng.logseries(one_minus_gamma))

        if self.eta > 0.0 and self.gamma**self.eta <= 0.5:
            k = 0
            while k == 0:
                k = int(rng.negative_binomial(self.eta, self.gamma))
            return k

        probability = (
            self.eta
            * one_minus_gamma
            / math.expm1(-self.eta * math.log(self.gamma))
        )
        cumulative_probability = probability
        uniform = float(rng.random())
        k = 1

        while uniform >= cumulative_probability:
            probability *= (
                (self.eta + k) * one_minus_gamma / (k + 1.0)
            )
            k += 1
            next_cumulative_probability = cumulative_probability + probability

            if next_cumulative_probability == cumulative_probability:
                raise ArithmeticError(
                    "The TNB tail became too small to sample numerically."
                )

            cumulative_probability = next_cumulative_probability

        return k

    def sample_conditional(
        self,
        m: int,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Draw and return K0 conditioned on K0 >= m."""
        m = _validate_positive_integer(m, "m")

        if m == 1:
            return self.sample(rng)
        if rng is None:
            rng = np.random.default_rng()

        probability = self.pmf(m) / self.survival_probability(m)
        cumulative_probability = probability
        uniform = float(rng.random())
        k = m
        one_minus_gamma = 1.0 - self.gamma

        while uniform >= cumulative_probability:
            probability *= (
                (self.eta + k) * one_minus_gamma / (k + 1.0)
            )
            k += 1
            next_cumulative_probability = cumulative_probability + probability

            if next_cumulative_probability == cumulative_probability:
                raise ArithmeticError(
                    "The conditional TNB tail became too small to sample "
                    "numerically."
                )

            cumulative_probability = next_cumulative_probability

        return k

    def log_expected_binomial(self, m: int) -> float:
        r"""
        Return

            log E[binom(K, m)],

        where K = K0 | K0 >= m.
        """
        m = _validate_positive_integer(m, "m")
        survival = self.survival_probability(m)

        log_value = (
            (1.0 - m) * math.log(self.gamma)
            + math.log(self.mean())
            + (m - 1.0) * math.log1p(-self.gamma)
            - math.lgamma(m + 1)
            - math.log(survival)
        )

        for i in range(1, m):
            log_value += math.log(self.eta + i)

        return log_value
