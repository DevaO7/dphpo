import math
import numpy as np
from scipy.special import gammaln, logsumexp
from dp_accounting import dp_event
from dp_accounting.rdp import RdpAccountant
import dp_accounting
import math
from scipy.optimize import brentq


def logcomb(n, k):
    """Returns the logarithm of comb(n,k)"""
    return gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)


def RDP_epsilon_bound_gaussian(alpha, sigma_gaussian_actual):
    """Returns the epsilon RDP bound for Gaussian mechanism with std parameter sigma_gaussian_actual"""
    return 0.5 * alpha / (sigma_gaussian_actual ** 2)


def cgf_subsampling_for_int_alpha(alpha: int, eps_func, sub_ratio, K=None, s=None, sigma_gaussian=None, numerical=False):
    """
    Parameters:
    :param alpha: int, >1
    :param eps_func: fun(float->float), epsilon RDP bound evaluation function
    :param sub_ratio: subsampling ratio

    Returns a tight upper bound of the CGF(alpha) for the s-subsampled eps_func(alpha),
    ie (alpha-1)*eps_subsampled(alpha)
    """
    if s is None:  # When eps function is RDP Gaussian mechanism
        alpha = int(alpha)
        log_moment_two = 2 * np.log(sub_ratio) + logcomb(alpha, 2) + np.minimum(
            np.log(4) + eps_func(2., sigma_gaussian) + np.log(1 - np.exp(-eps_func(2., sigma_gaussian))),
            eps_func(2., sigma_gaussian) + np.log(2)
        )
        log_moment_j = lambda j: np.log(2) + (j - 1) * eps_func(j, sigma_gaussian) + j * np.log(sub_ratio) + logcomb(alpha, j)
        all_log_moments_j = [log_moment_j(j) for j in range(3, alpha + 1)]
    else:  # When eps function is intermediate
        alpha = int(alpha)
        log_moment_two = 2 * np.log(sub_ratio) + logcomb(alpha, 2) + np.minimum(
            np.log(4) + eps_func(2., K, s, sigma_gaussian, numerical) + np.log(1 - np.exp(-eps_func(2., K, s, sigma_gaussian, numerical))),
            eps_func(2., K, s, sigma_gaussian, numerical) + np.log(2)
        )
        log_moment_j = lambda j: np.log(2) + (j - 1) * eps_func(j, K, s, sigma_gaussian, numerical) + j * np.log(sub_ratio) + logcomb(alpha, j)
        all_log_moments_j = [log_moment_j(j) for j in range(3, alpha + 1)]

    return logsumexp([0, log_moment_two] + all_log_moments_j)

def numerical_rdp_accounting(alpha: int, K: int, s: float, sigma: float):
    """
    Parameters:
    :param alpha: int, >1
    :param K: nb of compositions
    :param s: subsampling ratio
    :param sigma: standard deviation of Gaussian noise

    Returns an upper RDP epsilon bound after K composed s-subsampled Gaussian mechanisms.
    """
    single_step = dp_accounting.dp_event.PoissonSampledDpEvent(
        sampling_probability=s,
        event=dp_event.GaussianDpEvent(noise_multiplier=sigma)
    )
    total_event = dp_event.SelfComposedDpEvent(single_step, K)
    order = [alpha]
    accountant = RdpAccountant(order)
    accountant.compose(total_event)
    return accountant._rdp[0]


def intermediate_epsilon_rdp_bound_for_int_alpha(alpha: int, K, s, sigma_gaussian_actual, numerical=False):
    """
    Parameters:
    :param alpha: int, >1

    Returns an upper RDP epsilon bound after K composed s-subsampled Gaussian mechanisms.
    """
    if numerical:
        subsampled_gaussian_rdp = numerical_rdp_accounting(alpha, K, s, sigma_gaussian_actual)
        gaussian_rdp = K*RDP_epsilon_bound_gaussian(alpha, sigma_gaussian_actual)
        return min(subsampled_gaussian_rdp, gaussian_rdp)
    else: 
        subsampled_gaussian_rdp = K * cgf_subsampling_for_int_alpha(
        alpha, RDP_epsilon_bound_gaussian, s, K=None, s=None, sigma_gaussian=sigma_gaussian_actual
    ) / (alpha - 1)
        gaussian_rdp = K*RDP_epsilon_bound_gaussian(alpha, sigma_gaussian_actual)
        return min(subsampled_gaussian_rdp, gaussian_rdp)


def epsilon_rdp_bound_for_int_alpha(alpha: int, T, K, l, s, sigma_gaussian_actual, numerical=False):
    """
    Parameters:
    :param alpha: int, >1

    Returns an upper RDP epsilon bound after T composed l-subsampled [K composed s-subsampled Gaussian mechanisms].
    """
    subsampled_eps = cgf_subsampling_for_int_alpha(
        alpha, intermediate_epsilon_rdp_bound_for_int_alpha, l, K, s, sigma_gaussian_actual, numerical
    ) / (alpha - 1)
    eps = intermediate_epsilon_rdp_bound_for_int_alpha(alpha, K, s, sigma_gaussian_actual, numerical)
    return T * min(subsampled_eps, eps)

def epsilon_rdp_bound_for_float_alpha(alpha: float, T, K, l, s, sigma_gaussian_actual, numerical):
    """
    Parameters:
    :param alpha: float, >1

    Returns an upper RDP epsilon bound after T composed l-subsampled [K composed s-subsampled Gaussian mechanisms],
    using linear interpolation on the CGF (by convexity) to approximate the bound.
    """
    floor_alpha = math.floor(alpha)
    ceil_alpha = math.ceil(alpha)

    if floor_alpha == 1:
        first = 0.
    else:
        first = (1 - alpha + floor_alpha) * (floor_alpha - 1) * epsilon_rdp_bound_for_int_alpha(
            floor_alpha, T, K, l, s, sigma_gaussian_actual, numerical
        ) / (alpha - 1)

    second = (alpha - floor_alpha) * (ceil_alpha - 1) * epsilon_rdp_bound_for_int_alpha(
        ceil_alpha, T, K, l, s, sigma_gaussian_actual, numerical
    ) / (alpha - 1)

    return first + second


def epsilon_dp_bound_for_int_alpha(alpha: int, delta: float, T, K, l, s, sigma_gaussian_actual, numerical=False):
    """
    Parameters:
    :param alpha: int, >1

    Returns an upper DP epsilon bound after T composed l-susampled [K composed s-subsampled Gaussian mechanisms].
    """
    return epsilon_rdp_bound_for_int_alpha(alpha, T, K, l, s, sigma_gaussian_actual, numerical) + np.log(1 / delta) / (alpha - 1)


def epsilon_dp_bound_for_float_alpha(alpha: float, T, K, l, s, delta, sigma_gaussian_actual, numerical = False):
    """
    Parameters:
    :param alpha: float, >1

    Returns an upper DP epsilon bound after T composed l-susampled [K composed s-subsampled Gaussian mechanisms].
    """
    return epsilon_rdp_bound_for_float_alpha(alpha, T, K, l, s, sigma_gaussian_actual, numerical) + np.log(1 / delta) / (alpha - 1)



def expected_K_trunc_nb(eta: float, gamma: float) -> float:
    if not (-1 < eta):
        raise ValueError("eta must satisfy eta > -1")
    if not (0 < gamma < 1):
        raise ValueError("gamma must satisfy 0 < gamma < 1")
    if eta == 0:
        return (1.0 / gamma - 1.0) / math.log(1.0 / gamma)
    return eta * (1.0 - gamma) / (gamma * (1.0 - gamma ** eta))

def theorem2_rdp_bound(
    lambda_order: float,
    lambda_hat: float,
    eta: float,
    gamma: float,
    expected_K: float,
    eps_lambda: float,
    eps_lambda_hat: float | None,
) -> float:
    if lambda_order <= 1:
        raise ValueError("lambda_order must be > 1")
    if lambda_hat < 1:
        raise ValueError("lambda_hat must be >= 1")

    term = (
        eps_lambda
        + ((1.0 + eta) * math.log(1.0 / gamma)) / lambda_hat
        + math.log(expected_K) / (lambda_order - 1.0)
    )

    if lambda_hat > 1:
        term += (1.0 + eta) * (1.0 - 1.0 / lambda_hat) * eps_lambda_hat

    return term

def hp_epsilon_dp_bound(
    T, K, M, R, l, s, sigma_gaussian_actual, delta,
    lambda_order, lambda_hat, eta, gamma, numerical,
    eps_fn_lambda, eps_fn_lambda_hat
):
    eps_lambda = eps_fn_lambda(
        lambda_order, T, K, l, s, sigma_gaussian_actual, numerical
    )

    eps_lambda_hat = None
    if lambda_hat > 1:
        eps_lambda_hat = eps_fn_lambda_hat(
            lambda_hat, T, K, l, s, sigma_gaussian_actual, numerical
        )

    EK = expected_K_trunc_nb(eta, gamma)

    eps_rdp = theorem2_rdp_bound(
        lambda_order=lambda_order,
        lambda_hat=lambda_hat,
        eta=eta,
        gamma=gamma,
        expected_K=EK,
        eps_lambda=eps_lambda,
        eps_lambda_hat=eps_lambda_hat,
    )
    return eps_rdp + math.log(1.0 / delta) / (lambda_order - 1.0)

def gamma_from_expected_K_logarithmic(K_target: float) -> float:
    if K_target <= 1:
        raise ValueError("Need K_target > 1 since K >= 1 almost surely.")
    
    def f(gamma):
        return ((1.0 / gamma) - 1.0) / math.log(1.0 / gamma) - K_target
    
    return brentq(f, 1e-12, 1 - 1e-12)

def compute_hp_epsilon(
    T, K, M, R, l, s, sigma_gaussian, eta, expected_K, numerical,
    lambda_int_max=100, lambda_float_points=1000
):
    delta = 1.0 / (M * R)
    sigma_gaussian_actual = sigma_gaussian * math.sqrt(l * M)
    gamma = gamma_from_expected_K_logarithmic(expected_K)
    # Integer grid for lambda_hat, including 1 as allowed by theorem.
    lambda_hat_grid = np.arange(1, lambda_int_max + 1)

    # Global float grid for lambda, not just local refinement.
    lambda_grid = np.linspace(1.0001, float(lambda_int_max), lambda_float_points)

    best_eps = float("inf")
    best_lambda = None
    best_lambda_hat = None

    

    for lambda_order in lambda_grid:
        for lambda_hat in lambda_hat_grid:
            print(f"Evaluating for lambda={lambda_order:.4f}, lambda_hat={lambda_hat}")
            eps = hp_epsilon_dp_bound(
                T=T, K=K, M=M, R=R, l=l, s=s,
                sigma_gaussian_actual=sigma_gaussian_actual,
                delta=delta,
                lambda_order=float(lambda_order),
                lambda_hat=float(lambda_hat),
                eta=eta, gamma=gamma, numerical=numerical,
                eps_fn_lambda=epsilon_rdp_bound_for_float_alpha,
                eps_fn_lambda_hat=epsilon_rdp_bound_for_int_alpha,
            )
            if eps < best_eps:
                best_eps = eps
                best_lambda = float(lambda_order)
                best_lambda_hat = float(lambda_hat)

    return best_eps

if __name__ == "__main__":
    T = 300
    K = 50
    M = 40
    R = 2000
    l = 0.2
    s = 0.2
    sigma_gaussian = 20
    eta = 0
    gamma = gamma_from_expected_K_logarithmic(10)
    numerical = False

    best_eps, best_lambda, best_lambda_hat = compute_hp_epsilon(
        T=T, K=K, M=M, R=R, l=l, s=s, sigma_gaussian=sigma_gaussian,
        eta=eta, expected_K=expected_K, numerical=numerical,
        lambda_int_max=100, lambda_float_points=1000
    )

    print(f"Best epsilon: {best_eps}")
    print(f"Best lambda: {best_lambda}")
    print(f"Best lambda_hat: {best_lambda_hat}")