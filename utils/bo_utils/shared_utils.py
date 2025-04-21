import os
import math
import warnings
import re
from dataclasses import dataclass
import matplotlib.pyplot as plt

import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


import os
import torch
from botorch.utils.transforms import unnormalize

def load_top_optimization_results(
    results_dir_dict,
    stim_bounds,
    prefix='run_',
    verbose=False
):
    x_dict, y_dict = {}, {}

    for method, result_dir in results_dir_dict.items():
        try:
            if verbose:
                print(f"Method: {method}")
            folder = get_most_recent_result_subfolder_name(result_dir=result_dir, prefix=prefix)
            updated_x = torch.load(os.path.join(folder, "updated_x.pt"))
            updated_y = torch.load(os.path.join(folder, "updated_y.pt"))

            if method == 'turbo':
                updated_x = unnormalize(updated_x, stim_bounds)

            x_dict[method] = updated_x
            y_dict[method] = updated_y

        except Exception as e:
            if verbose:
                print(f"Failed to load results for method '{method}': {e}")

    return x_dict, y_dict



def plot_top_param_distributions(space, x_dict, y_dict, num_top_points=20, indices_to_plot=None):
    """
    Plot the distribution of top-performing parameter values for each method.

    Args:
        space (list): List of skopt.space.Real objects, each with a `.name` attribute.
        stim_bounds (torch.Tensor): Tensor of shape (2, D) with min and max bounds for each parameter.
        x_dict (dict): Maps method names to torch.Tensor of evaluated parameter values (shape: N x D).
        y_dict (dict): Maps method names to torch.Tensor of objective values (shape: N).
        num_top_points (int): Number of top-performing trials to show per method.
    """
    index_to_params = {i: dim.name for i, dim in enumerate(space)}
    stim_bounds = torch.tensor([[dim.low, dim.high] for dim in space], dtype=torch.float).T

    methods_to_y_axis = {
        'turbo': 3,
        'ibnn': 2,
        'baseline': 1
    }

    if indices_to_plot is None:
        indices_to_plot = range(len(space))

    for index in indices_to_plot:
        param_name = index_to_params[index]
        bounds = stim_bounds[:, index]
        print(f"{param_name}: {bounds}")

        fig, ax = plt.subplots()

        for method, x_vals in x_dict.items():
            y_vals = y_dict[method]
            top_indices = torch.topk(y_vals.squeeze(), num_top_points).indices
            top_params = x_vals[top_indices]
            top_param_values = top_params[:, index]
            y_level = methods_to_y_axis[method]

            sorted_extremes = torch.sort(top_param_values[[0, -1]]).values.tolist()
            print(f"Method: {method}, extremes: {sorted_extremes}")

            ax.plot(top_param_values, [y_level] * num_top_points, 'o', markersize=5, label=method)

        ax.set_xlim(bounds.tolist())
        ax.set_title(f"{param_name}: {bounds.tolist()}")
        ax.set_yticks(list(methods_to_y_axis.values()))
        ax.set_yticklabels(list(methods_to_y_axis.keys()))
        ax.set_xlabel(param_name)
        ax.set_ylabel("Method")
        ax.legend()
        plt.show()



def plot_best_results(updated_x, updated_y, objective_func_tensor, num_top_points=20):
    # find the index of top k in updated_y
    top_5_indices = torch.topk(updated_y.squeeze(), num_top_points).indices
    # get the corresponding parameters
    top_5_params = updated_x[top_5_indices]
    # get the corresponding values
    top_5_values = updated_y[top_5_indices]

    for i in range(num_top_points):
        params = top_5_params[i]
        print(f"Top {i+1} parameters: {params}")
        print(f"Top {i+1} values: {top_5_values[i]}")
        # run the objective function with these parameters
        objective_func_tensor(params)
    return


def _get_most_recent_result_subfolder_number(result_dir='all_stored_results/ibnn_results', prefix='run_'):
    # Get all folders in base_dir that match the pattern run_X
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(
            f"The directory '{result_dir}' does not exist. Please create it before running the script.")
    else:
        existing = [d for d in os.listdir(
            result_dir) if os.path.isdir(os.path.join(result_dir, d))]

    # Get all folders that run_X
    run_numbers = []

    # Extract numbers using regex
    for name in existing:
        match = re.match(rf'{re.escape(prefix)}(\d+)', name)
        if match:
            run_numbers.append(int(match.group(1)))

    # Determine most recent folder number
    most_recent_run = max(run_numbers, default=0)
    return most_recent_run


def get_most_recent_result_subfolder_name(result_dir='all_stored_results/ibnn_results', prefix='run_'):
    most_recent_run = _get_most_recent_result_subfolder_number(
        result_dir, prefix=prefix)
    most_recent_folder = os.path.join(
        result_dir, f'{prefix}{most_recent_run}')
    return most_recent_folder


def get_new_result_subfolder(result_dir='all_stored_results/ibnn_results', prefix='run_'):
    most_recent_run = _get_most_recent_result_subfolder_number(
        result_dir, prefix=prefix)

    # Determine next folder number
    next_run = most_recent_run + 1
    new_folder = os.path.join(result_dir, f'{prefix}{next_run}')
    os.makedirs(new_folder)

    print(f'Created new subfolder for storing results: {new_folder}')
    return new_folder


def save_results_at_fixed_iterations(iteration, train_x, train_y, result_dir, result_folder):
    if iteration == 10:
        # change the result folder to the new one
        result_folder = get_new_result_subfolder(result_dir)

    if iteration % 1 == 0:
        print(f'Saved updated_x.pt and updated_y.pt in the folder {result_folder}')
        print('Number of training points:', train_x.size(0))
        torch.save(train_x, os.path.join(result_folder, "updated_x.pt"))
        torch.save(train_y, os.path.join(result_folder, "updated_y.pt"))

    if (iteration % 10 == 0) & (iteration >= 10):
        # save a backup
        print(f'Saved updated_x_backup.pt and updated_y_backup.pt in the folder {result_folder}')
        torch.save(train_x, os.path.join(
            result_folder, "updated_x_backup.pt"))
        torch.save(train_y, os.path.join(
            result_folder, "updated_y_backup.pt"))
    return result_folder
