import os
import math
import warnings
import re
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
            folder = get_latest_result_subfolder_name(
                result_dir=result_dir, prefix=prefix)
            updated_x = torch.load(os.path.join(folder, "updated_x.pt"))
            updated_y = torch.load(os.path.join(folder, "updated_y.pt"))

            x_dict[method] = updated_x
            y_dict[method] = updated_y

        except Exception as e:
            if verbose:
                print(f"Failed to load results for method '{method}': {e}")

    return x_dict, y_dict


def plot_top_param_distributions(space, x_dict, y_dict, num_top_points=5, indices_to_plot=None):
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
    stim_bounds = torch.tensor([[dim.low, dim.high]
                               for dim in space], dtype=torch.float).T

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

            sorted_extremes = torch.sort(
                top_param_values[[0, -1]]).values.tolist()
            print(f"Method: {method}, extremes: {sorted_extremes}")

            ax.plot(top_param_values, [
                    y_level] * num_top_points, 'o', markersize=5, label=method)

        ax.set_xlim(bounds.tolist())
        ax.set_title(f"{param_name}: {bounds.tolist()}")
        ax.set_yticks(list(methods_to_y_axis.values()))
        ax.set_yticklabels(list(methods_to_y_axis.keys()))
        ax.set_xlabel(param_name)
        ax.set_ylabel("Method")
        ax.legend()
        plt.show()


def plot_best_results(updated_x, updated_y, objective_func_tensor, num_top_points=5,
                      fr_window=[1100, 6100]):
    output_unnorm_factor = fr_window[1] - fr_window[0]
    print('Note: the output is unnormalized by multiplying by the factor:',
          output_unnorm_factor)
    # find the index of top k in updated_y
    top_n_indices = torch.topk(updated_y.squeeze(), num_top_points).indices
    # get the corresponding parameters
    top_n_params = updated_x[top_n_indices]
    # get the corresponding values
    top_n_values = updated_y[top_n_indices] * output_unnorm_factor

    for i in range(num_top_points):
        params = top_n_params[i]
        print(f"Top {i+1} parameters: {params}")
        print(f"Top {i+1} values: {top_n_values[i]}")
        # run the objective function with these parameters
        objective_func_tensor(params, plotting=True)
    return


def _get_latest_result_subfolder_number(result_dir='all_stored_results/ibnn_results', prefix='run_'):
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
    latest_run = max(run_numbers, default=0)
    return latest_run


def get_latest_result_subfolder_name(result_dir='all_stored_results/ibnn_results', prefix='run_'):
    latest_run = _get_latest_result_subfolder_number(
        result_dir, prefix=prefix)
    latest_folder = os.path.join(
        result_dir, f'{prefix}{latest_run}')
    return latest_folder


def get_new_result_subfolder(result_dir='all_stored_results/ibnn_results', prefix='run_'):
    os.makedirs(result_dir, exist_ok=True)

    latest_run = _get_latest_result_subfolder_number(
        result_dir, prefix=prefix)

    # Determine next folder number
    next_run = latest_run + 1
    new_folder = os.path.join(result_dir, f'{prefix}{next_run}')
    os.makedirs(new_folder)

    print(f'Created new subfolder for storing results: {new_folder}')
    return new_folder


def save_results_at_fixed_iterations(iteration, train_x, train_y, result_dir, result_folder):
    if iteration == 10:
        # change the result folder to the new one
        result_folder = get_new_result_subfolder(result_dir)

    if iteration % 1 == 0:
        print(
            f'Saved updated_x.pt and updated_y.pt in the folder {result_folder}')
        print('Number of training points:', train_x.size(0))
        torch.save(train_x, os.path.join(result_folder, "updated_x.pt"))
        torch.save(train_y, os.path.join(result_folder, "updated_y.pt"))

    if (iteration % 10 == 0) & (iteration >= 10):
        # save a backup
        print(
            f'Saved updated_x_backup.pt and updated_y_backup.pt in the folder {result_folder}')
        torch.save(train_x, os.path.join(
            result_folder, "updated_x_backup.pt"))
        torch.save(train_y, os.path.join(
            result_folder, "updated_y_backup.pt"))
    return result_folder


def save_results(train_x, train_y, result_dir='all_stored_results/ibnn_results'):
    result_folder = get_new_result_subfolder(result_dir)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    torch.save(train_x, os.path.join(result_folder, "updated_x.pt"))
    torch.save(train_y, os.path.join(result_folder, "updated_y.pt"))
    return


def get_latest_results(result_dir, prefix='run_'):
    latest_folder = get_latest_result_subfolder_name(
        result_dir=result_dir, prefix=prefix)
    updated_x = torch.load(os.path.join(latest_folder, "updated_x.pt"))
    updated_y = torch.load(os.path.join(latest_folder, "updated_y.pt"))
    return updated_x, updated_y


def repeat_eval_top_params(updated_x, updated_y, objective_func_tensor, method,
                           num_top_points=5, num_repeats=5, fr_window=[1100, 6100]):
    """
    Re-evaluates the top-performing parameter sets multiple times using the objective function.

    Args:
        updated_x (Tensor): All evaluated input parameters.
        updated_y (Tensor): Corresponding objective values (to minimize).
        objective_func_tensor (callable): The objective function to evaluate.
        num_top_points (int): Number of top parameter sets to re-evaluate.
        fr_window (list): Firing rate window used to compute unnormalized outputs.
        num_repeats (int): Number of repeated evaluations for each top parameter set.

    Returns:
        all_best_params (List[Tensor]): Top parameter vectors.
        all_values_reps (List[List[float]]): Repeated objective values for each top parameter.
    """
    output_scale = fr_window[1] - fr_window[0]
    print('Note: output values will be unnormalized by factor:', output_scale)

    # Get indices of top-performing parameters
    top_indices = torch.topk(updated_y.squeeze(), num_top_points).indices
    top_params = updated_x[top_indices]
    top_values = updated_y[top_indices] * output_scale

    all_best_params = []
    all_values_reps = []

    for i in range(num_top_points):
        params = top_params[i]
        print(f"{method}: Top {i+1} parameters: {params}")
        print(f"{method}: Top {i+1} (unnormalized) value: {top_values[i].item():.4f}")
        all_best_params.append(params)

        repeated_values = []
        for _ in range(num_repeats):
            value = objective_func_tensor(params, plotting=True)
            repeated_values.append(value)

        all_values_reps.append(repeated_values)

    return all_best_params, all_values_reps



def build_repeated_eval_df(y_rep_dict, save_path=None):
    """
    Convert repeated objective values into a tidy DataFrame for analysis/plotting.

    Args:
        y_rep_dict (dict): Maps each method name to a list of lists of repeated torch tensors.
                           Each sublist contains multiple objective values for a top-k point.

    Returns:
        pd.DataFrame: Tidy DataFrame with columns ['top_k', 'y', 'method']
    """

    # make sure y_rep_dict is not empty
    if not y_rep_dict:
        raise ValueError(
            "y_rep_dict is empty. Please provide a non-empty dictionary.")

    repeated_eval_df = pd.DataFrame([])
    # for the top k points for each method, average the multiple sampled values
    for method, _ in y_rep_dict.items():
        y_vals = [torch.stack(tensors).numpy()
                  for tensors in y_rep_dict[method]]
        y_vals_flat = np.array(y_vals).reshape(-1)

        # get the corresponding top-k indices
        num_top_points = len(y_vals)
        num_repeats = len(y_vals[0])
        top_k = np.repeat(np.arange(1, num_top_points+1), num_repeats)

        # Convert to DataFrame
        temp_df_repeats = pd.DataFrame({'top_k': top_k, 'y': y_vals_flat})
        temp_df_repeats['method'] = method
        repeated_eval_df = pd.concat(
            [repeated_eval_df, temp_df_repeats], ignore_index=True)

    if save_path is not None:
        # make sure the directory (the parent of save_path) exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        repeated_eval_df.to_csv(save_path, index=False)

    return repeated_eval_df


def build_avg_eval_df(y_rep_dict, save_path=None):

    # make sure y_rep_dict is not empty
    if not y_rep_dict:
        raise ValueError(
            "y_rep_dict is empty. Please provide a non-empty dictionary.")

    all_averages = {}
    # for the top k points for each method, average the multiple sampled values
    for method, _ in y_rep_dict.items():
        averages = [torch.stack(tensors).mean().item()
                    for tensors in y_rep_dict[method]]
        all_averages[method] = np.array(averages)

    # Convert to DataFrame
    avg_eval_df = pd.DataFrame(all_averages)

    # Add a column for the rank (e.g., top-k index)
    avg_eval_df.index.name = "Top-K"
    avg_eval_df.reset_index(inplace=True)

    if save_path is not None:
        # make sure the directory (the parent of save_path) exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        avg_eval_df.to_csv(save_path, index=False)
    return avg_eval_df, all_averages


def plot_avg_eval_df(avg_eval_df,
                     all_methods=['baseline', 'turbo', 'ibnn', 'cma'],
                     fr_window=[1100, 6100],
                     top_n_to_plot=10,
                     num_repeats=5):
    """
    Plot the average evaluation DataFrame.

    Args:
        avg_eval_df (pd.DataFrame): DataFrame containing average evaluations.
        fr_window (list): Firing rate window used to compute unnormalized outputs.
        top_n_to_plot (int): Number of top parameter sets to plot.
        num_repeats (int): Number of repeats for the evaluation.
    """

    # Plot cumulative max or average values across top-k params
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in all_methods:
        if method not in avg_eval_df.columns:
            print(f"Method '{method}' not found in DataFrame columns.")
            continue
        # Adjust for the window
        y_averages = avg_eval_df[method].values * (fr_window[1] - fr_window[0])
        y_averages = y_averages[:top_n_to_plot]  # Limit to top-k params
        ax.plot(
            # 1-based indexing for top-k clarity
            range(1, len(y_averages) + 1),
            y_averages,
            label=method,
            linewidth=2,
            alpha=0.8,
            marker='o',
            markersize=4
        )

    # Set integer x-ticks
    ax.set_xticks(range(1, top_n_to_plot+1))

    # Labels and title
    ax.set_xlabel("Top-K Parameter Sets", fontsize=16)
    ax.set_ylabel("Mean Persistent Activity (ms)", fontsize=18)
    ax.set_ylim(0, fr_window[1] - fr_window[0])
    ax.set_title(
        f"Objective Value for Top Parameters (Average of {num_repeats} Repeats)", fontsize=20, pad=20)

    # Grid and legend
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title="Method", fontsize=15, title_fontsize=12)

    # Optional tweaks
    ax.tick_params(axis='both', labelsize=12)
    fig.tight_layout()
    plt.show()




