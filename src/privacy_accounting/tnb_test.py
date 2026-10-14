import math

import numpy as np

from privacy_accounting.rdp_utils import RdpCurve
from privacy_accounting.selection_accounting import compute_top_m_rdp
from privacy_accounting.tnb import TNBDistribution


def test_large_m_conditional_mean_calibration_avoids_tail_cancellation():
    distribution = TNBDistribution.from_conditional_mean(
        eta=0.0,
        m=50,
        target_mean=55.0,
    )

    assert math.isclose(
        distribution.conditional_mean(50),
        55.0,
        rel_tol=1e-11,
        abs_tol=1e-11,
    )
    assert math.isclose(
        distribution.survival_probability(50),
        math.fsum(distribution.pmf(k) for k in range(50, 1000)),
        rel_tol=1e-12,
        abs_tol=0.0,
    )


def test_large_m_top_m_accounting_returns_finite_curve():
    orders = np.arange(2, 101)
    base_curve = RdpCurve(
        orders=orders,
        epsilons=0.1 * orders,
    )

    result = compute_top_m_rdp(
        base_rdp_curve=base_curve,
        m=50,
        expected_num_trials=55.0,
        eta=0.0,
    )

    assert math.isclose(
        result.distribution.conditional_mean(50),
        55.0,
        rel_tol=1e-11,
        abs_tol=1e-11,
    )
    assert np.all(np.isfinite(result.rdp_curve.epsilons))
    assert math.isfinite(result.log_expected_binomial)


def test_log_expected_binomial_remains_finite_for_underflowing_tail():
    distribution = TNBDistribution(
        eta=0.0,
        gamma=1.0 - 1e-10,
    )

    assert math.isclose(
        distribution.conditional_mean(50),
        50.00000000009805,
        rel_tol=1e-14,
        abs_tol=1e-14,
    )
    assert math.isfinite(distribution.log_expected_binomial(50))
