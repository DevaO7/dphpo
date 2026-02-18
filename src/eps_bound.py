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
        return numerical_rdp_accounting(alpha, K, s, sigma_gaussian_actual)
    else: 
        return K * cgf_subsampling_for_int_alpha(
        alpha, RDP_epsilon_bound_gaussian, s, K=None, s=None, sigma_gaussian=sigma_gaussian_actual
    ) / (alpha - 1)


def epsilon_rdp_bound_for_int_alpha(alpha: int, T, K, l, s, sigma_gaussian_actual, numerical=False):
    """
    Parameters:
    :param alpha: int, >1

    Returns an upper RDP epsilon bound after T composed l-subsampled [K composed s-subsampled Gaussian mechanisms].
    """
    return T * cgf_subsampling_for_int_alpha(
        alpha, intermediate_epsilon_rdp_bound_for_int_alpha, l, K, s, sigma_gaussian_actual, numerical
    ) / (alpha - 1)


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
    return epsilon_dp_bound_for_float_alpha(alpha_float_min, T, K, l, s, delta, sigma_gaussian_actual, numerical=True)


def plot_comparison(parameter_varied, T, sigma, K, M, R, l, s):
    eps_num = []
    eps_theo = []
    if parameter_varied == 'sigma':
        parameter = np.arange(1, 1000, 10)
        for sigma in parameter:
            print(f"Computing for sigma={sigma}")
            # eps_num.append(compute_epsilon_numerical(T, K, M, R, l, s, sigma))
            eps_theo.append(compute_epsilon_theory(T, K, M, R, l, s, sigma))
    elif parameter_varied == 'T':
        parameter = np.arange(1, 1000, 25)
        for T in parameter:
            print(f"Computing for T={T}")
            # eps_num.append(compute_epsilon_numerical(T, K, M, R, l, s, sigma))
            eps_theo.append(compute_epsilon_theory(T, K, M, R, l, s, sigma))
    elif parameter_varied == 'K':
        parameter = np.arange(1, 100, 1)
        for K in parameter:
            print(f"Computing for K={K}")
            # eps_num.append(compute_epsilon_numerical(T, int(K), M, R, l, s, sigma))
            eps_theo.append(compute_epsilon_theory(T, K, M, R, l, s, sigma))
    elif parameter_varied == 'l':
        parameter = np.arange(0.01, 1, 0.01)
        for l in parameter:
            print(f"Computing for l={l}")
            # eps_num.append(compute_epsilon_numerical(T, K, M, R, l, s, sigma))
            eps_theo.append(compute_epsilon_theory(T, K, M, R, l, s, sigma))
    elif parameter_varied == 's':
        parameter = np.arange(0.01, 1, 0.01)
        for s in parameter:
            print(f"Computing for s={s}")
            # eps_num.append(compute_epsilon_numerical(T, K, M, R, l, s, sigma))
            eps_theo.append(compute_epsilon_theory(T, K, M, R, l, s, sigma))

    # plt.plot(parameter, eps_num, label='Numerical Accounting')
    plt.plot(parameter, eps_theo, label='Theoretical Accounting')
    plt.ylabel('Epsilon')
    plt.xlabel(f'{parameter_varied}')
    plt.title(f'Epsilon vs {parameter_varied}')
    plt.legend()
    plt.savefig(f'comparison_epsilon_{parameter_varied}_2.png')
    
if __name__ == "__main__":
    T = 500
    sigma = 5
    K = 50
    M = 40
    R = 2000
    l = 0.21
    s = 0.2
    delta = 1 / (M * R)
    eps_num = compute_epsilon_numerical(T, K, M, R, l, s, sigma)
    print(f"Epsilon (Numerical Accounting): {eps_num:.2f}")


    # T_values = np.arange(2, 1000, 25)
    # l_values = np.arange(0.01, 1, 0.01)
    # eps_values_num = []
    # with open('epsilon_values_numerical_T_l.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['T', 'l', 'Epsilon Numerical'])

    # for T in T_values:
    #     for l in l_values:
    #         eps_num = round(compute_epsilon_numerical(T=int(T), K=int(K), M=100, R=int(0.8*5000), l=l, s=0.2, sigma_gaussian=int(sigma)), 2)
    #         with open('epsilon_values_numerical_T_l.csv', 'a') as f:
    #             writer = csv.writer(f)
    #             writer.writerow([int(T), l, eps_num])
    #         eps_values_num.append(eps_num)
    
    # # 3d plot
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # T_grid, l_grid = np.meshgrid(T_values, l_values)
    # eps_grid = np.array(eps_values_num).reshape(len(l_values), len(T_values))
    # ax.plot_surface(T_grid, l_grid, eps_grid, cmap='viridis')
    # ax.set_xlabel('T')
    # ax.set_ylabel('l')
    # ax.set_zlabel('Epsilon')
    # plt.title('Epsilon DP Bound vs T and l (Numerical Accounting)')
    # plt.savefig('epsilon_dp_bound_vs_T_l_numerical.png')


    # plot_comparison('T', T, sigma, K, M, R, l, s)










# # ... (previous code)


# table = []

# x = []
# y = []

# # for sigma in np.arange(1, 1000, 10):
# # for T in np.arange(1, 1000, 5):

# for sigma in np.logspace(1, 10, 100):
#     eps_theory = round(compute_epsilon_theory(T=int(250), K=5, M=100, R=int(0.8*5000), l=0.2, s=0.2, sigma_gaussian=int(sigma)), 4)
#     eps_numerical = round(compute_epsilon_numerical(T=int(250), K=5, M=100, R=int(0.8*5000), l=0.2, s=0.2, sigma_gaussian=int(sigma)), 4)
#     temp = {
#         "T": int(250),
#         "K": 5,
#         "M": 100,
#         "R": int(0.8 * 2500),
#         "l": 0.2,
#         "s": 0.2,
#         "sigma_gaussian": int(sigma),
#         "epsilon": float(eps)
#     }
#     table.append(temp) 
#     x.append(int(sigma))
#     y.append(eps)
#     print(f"sigma: {int(sigma)}, Epsilon: {eps}")

# # 1. Define the directory path you want to save results in
# results_dir = 'Differential-Privacy-for-Heterogeneous-Federated-Learning/results/'

# # 2. Create the directory if it doesn't exist
# os.makedirs(results_dir, exist_ok=True)

# # 3. Now, save your files into that directory
# json_path = os.path.join(results_dir, 'eps_T_alpha.json')
# with open(json_path, 'w') as f:
#     json.dump(table, f, indent=4)

# image_path = os.path.join(results_dir, 'epsilon_dp_bound_vs_T_new_setting_large_T.png')
# plt.plot(x, y)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('sigma')
# plt.ylabel('Epsilon (DP Bound)')
# plt.title('Epsilon DP Bound vs T')
# plt.grid()
# plt.savefig(image_path)

# print(f"\nResults saved successfully in '{results_dir}'")


# # for T in range(1, 1001):
# # for l in np.arange(0, 1, 0.01):
# # for K in range(1, 100):
# # for s in np.arange(0, 1, 0.01):
# # for sigma in np.arange(1, 1000, 10):