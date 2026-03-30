import numpy as np
from scipy.special import gammaln, logsumexp
import math
import json
import matplotlib.pyplot as plt
import os
from dp_accounting import dp_event
from dp_accounting.rdp import RdpAccountant
import dp_accounting
import csv


# Meta parameters
# T: nb of communication rounds
# K: nb of local updates
# M: nb of users
# R: nb of data points used for training
# The privacy parameter epsilon is calculated for any third party who has access to the last iterate of the algorithm
# Our method consists of a minimization problem over the variable `alpha` from the RDP bound
# We notably use the upper bound for subsampling provided in Theorem 9 in https://arxiv.org/pdf/1808.00087.pdf

# Remark that our framework is only available for mechanisms with eps(infinity)=+inf !
# (verified for Gaussian mechanisms and its compositions)


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


def compute_epsilon_theory(T, K, M, R, l, s, sigma_gaussian):
    # Parameters to tune by hand:
    # alpha_int_max: int
    # n_points: int
    delta = 1 / (M * R)
    sigma_gaussian_actual = sigma_gaussian * np.sqrt(l * M)

    # 1. Determine the integer alpha with the best DP bound (grid search between 2 and alpha_int_max)
    alpha_int_max = 100
    alpha_int_space = np.arange(2, alpha_int_max + 1)
    argmin_int = np.argmin([
        epsilon_dp_bound_for_int_alpha(alpha_int, delta, T, K, l, s, sigma_gaussian_actual)
        for alpha_int in alpha_int_space
    ])
    alpha_int_min = alpha_int_space[argmin_int]
    if alpha_int_min == alpha_int_max:
        print("Increase alpha_int_max!")

    alpha_lower = alpha_int_min - 1. + 0.0001  # instability around alpha=1
    alpha_upper = alpha_int_min + 1.

    # 2. Determine the float alpha with the best DP bound (grid search around alpha_int_min: +-1)
    n_points = 1000  # precision of the grid
    alpha_float_space = np.linspace(alpha_lower, alpha_upper, n_points)
    idx_min = np.argmin([
        epsilon_dp_bound_for_float_alpha(alpha_float, T, K, l, s, delta, sigma_gaussian_actual)
        for alpha_float in alpha_float_space
    ])
    alpha_float_min = alpha_float_space[idx_min]
    return epsilon_dp_bound_for_float_alpha(alpha_float_min, T, K, l, s, delta, sigma_gaussian_actual)

def compute_epsilon_numerical(T, K, M, R, l, s, sigma_gaussian):
    # Parameters to tune by hand:
    # alpha_int_max: int
    # n_points: int
    delta = 1 / (M * R)
    # sigma_gaussian_actual = sigma_gaussian 
    sigma_gaussian_actual = sigma_gaussian * np.sqrt(l * M)

    # 1. Determine the integer alpha with the best DP bound (grid search between 2 and alpha_int_max)
    alpha_int_max = 100
    alpha_int_space = np.arange(2, alpha_int_max + 1)
    argmin_int = np.argmin([
        epsilon_dp_bound_for_int_alpha(alpha_int, delta, T, K, l, s, sigma_gaussian_actual, numerical=True)
        for alpha_int in alpha_int_space
    ])
    alpha_int_min = alpha_int_space[argmin_int]
    if alpha_int_min == alpha_int_max:
        print("Increase alpha_int_max!")

    alpha_lower = alpha_int_min - 1. + 0.0001  # instability around alpha=1
    alpha_upper = alpha_int_min + 1.

    # 2. Determine the float alpha with the best DP bound (grid search around alpha_int_min: +-1)
    n_points = 1000  # precision of the grid
    alpha_float_space = np.linspace(alpha_lower, alpha_upper, n_points)
    idx_min = np.argmin([
        epsilon_dp_bound_for_float_alpha(alpha_float, T, K, l, s, delta, sigma_gaussian_actual, numerical=True)
        for alpha_float in alpha_float_space
    ])
    alpha_float_min = alpha_float_space[idx_min]
    print(f"Best integer alpha: {alpha_int_min}")
    return epsilon_dp_bound_for_float_alpha(alpha_float_min, T, K, l, s, delta, sigma_gaussian_actual, numerical=True)


def hp_epsilon_rdp_bound(T, K, M, R, l, s, sigma_gaussian_actual, lambda_, 
                                         lambda_hat, eta, gamma, expected_K, epsilon_hat, epsilon_fn, numerical):
    epsilon_val = epsilon_fn(
        lambda_, T, K, M, R, l, s, sigma_gaussian_actual, numerical
    )

    epsilon_hat_value = epsilon_hat(lambda_hat, T, K, M, R, l, s, sigma_gaussian_actual, numerical)

    epsilon_prime = (
        epsilon_val
        + (1 + eta) * (1 - 1 / lambda_hat) * epsilon_hat_value
        + ((1 + eta) * math.log(1 / gamma)) / lambda_hat
        + math.log(expected_K) / (lambda_ - 1)
    )

    return epsilon_prime

def hp_epsilon_dp_bound_for_int_alpha(T, K, M, R, l, s, sigma_gaussian_actual, delta, lambda_int, lambda_hat_int, eta, gamma, expected_K, numerical, epsilon_hat=epsilon_rdp_bound_for_int_alpha, epsilon_fn=epsilon_rdp_bound_for_int_alpha):
    return hp_epsilon_rdp_bound(T, K, M, R, l, s, sigma_gaussian_actual, lambda_int, 
                                         lambda_hat_int, eta, gamma, expected_K, epsilon_hat, epsilon_fn, numerical) + np.log(1 / delta) / (lambda_int - 1)

def hp_epsilon_dp_bound_for_float_alpha(T, K, M, R, l, s, sigma_gaussian_actual, delta, lambda_float, lambda_hat_int, eta, gamma, expected_K, numerical, epsilon_hat=epsilon_rdp_bound_for_int_alpha, epsilon_fn=epsilon_rdp_bound_for_float_alpha):
    return hp_epsilon_rdp_bound(T, K, M, R, l, s, sigma_gaussian_actual, lambda_float,
                                         lambda_hat_int, eta, gamma, expected_K, epsilon_hat, epsilon_fn, numerical) + np.log(1 / delta) / (lambda_float - 1)

def compute_hp_epsilon(T, K, M, R, l, s, sigma_gaussian, eta, gamma, expected_K, numerical):
    delta = 1 / (M * R)
    sigma_gaussian_actual = sigma_gaussian * np.sqrt(l * M)
    lambda_int_max = 100
    lambda_int_space = np.arange(2, lambda_int_max + 1)
    lambda_hat_int_space = np.arange(1, lambda_int_max + 1)
    eps_lambda = np.full((len(lambda_int_space), len(lambda_hat_int_space)), np.inf)
    for lambda_int in lambda_int_space:
        for lambda_hat_int in lambda_hat_int_space:
            hp_epsilon = hp_epsilon_dp_bound_for_int_alpha(T, K, M, R, l, s, sigma_gaussian_actual, delta, lambda_int, lambda_hat_int, eta, gamma, expected_K, numerical)
            eps_lambda[lambda_int - 2, lambda_hat_int - 1] = hp_epsilon
    lambda_int_min, lambda_hat_int_min = np.unravel_index(np.argmin(eps_lambda), eps_lambda.shape)
    lambda_int_min += 2  
    lambda_hat_int_min += 1

    lambda_lower = (lambda_int_min) - 1. + 0.0001  # instability around alpha=1
    lambda_upper = lambda_int_min + 1.

    n_points = 1000  # precision of the grid
    lambda_float_space = np.linspace(lambda_lower, lambda_upper, n_points)
    idx_min = np.argmin([
        hp_epsilon_dp_bound_for_float_alpha(T, K, M, R, l, s, sigma_gaussian_actual, delta, lambda_float, lambda_hat_int_min, eta, gamma, expected_K, numerical)
        for lambda_float in lambda_float_space
    ])
    lambda_float_min = lambda_float_space[idx_min]
    return hp_epsilon_dp_bound_for_float_alpha(T, K, M, R, l, s, sigma_gaussian_actual, delta, lambda_float_min, lambda_hat_int_min, eta, gamma, expected_K, numerical)

if __name__ == "__main__":
    T = 500
    sigma = 20
    K = 50
    M = 40
    R = 2000
    l = 0.2
    s = 0.2
    eta = 0.1
    gamma = 0.1
    expected_K = 50
    compute_hp_epsilon(T, K, M, R, l, s, sigma, eta, gamma, expected_K, numerical=False)
    