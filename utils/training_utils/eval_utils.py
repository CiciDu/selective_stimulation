
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

from utils.methods_utils import ibnn_utils, shared_utils

def repeat_eval_top_params(updated_x, updated_y, objective_func_tensor, method,
                           num_top_points=5, num_repeats=5, fr_window=[1100, 8100],
                           iter_df_path=None):
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

    method_array = []
    y_value_array = []
    top_k_array = []
    num_iter_array = []
    index_array = []
    num_params = top_params.shape[1]
    param_dict = {f'param_{i+1}': [] for i in range(num_params)}

    all_best_params = []
    all_values_reps = []

    for i in range(num_top_points):
        params = top_params[i]

        repeated_values = []
        for j in range(num_repeats):
            print(
                f"{method}: Top {i+1} parameters — {params}; Unnormalized value: {top_values[i].item():.4f}")
            print(
                f"{method}: Repeating evaluation {j+1} of {num_repeats} for top {i+1} parameters.")
            value = objective_func_tensor(params, plotting=True)
            repeated_values.append(value)

            # we save the info for each repeated evaluation of each param set
            method_array.append(method)
            y_value_array.append(value.item())
            top_k_array.append(i+1)
            num_iter_array.append(j+1)
            index_array.append(top_indices[i].item())
            for k, v in enumerate(params):
                param_dict[f'param_{k+1}'].append(v.item())

        # after finish running the current param set, save the cumulated results into a dataframe
        iter_eval_df = pd.DataFrame({
            'method': method_array,
            'y_value': y_value_array,
            'top_k': top_k_array,
            'num_iter': num_iter_array,
            'index_in_result': index_array,
        })
        for k in range(num_params):
            iter_eval_df[f'param_{k+1}'] = param_dict[f'param_{k+1}']

        if iter_df_path is not None:
            os.makedirs(os.path.dirname(iter_df_path), exist_ok=True)
            iter_eval_df.to_csv(iter_df_path, index=False)
            print(f"Saved iter_eval_df to {iter_df_path}")

        # we also save the best params and their repeated values once for each param set
        all_best_params.append(params)
        all_values_reps.append(repeated_values)

    return all_best_params, all_values_reps, iter_eval_df


def make_iter_eval_df_from_dict(x_rep_dict, y_rep_dict, result_dir, method):
    
    updated_x, updated_y = shared_utils.get_latest_results(result_dir)

    method_array = []
    y_value_array = []
    top_k_array = []
    num_iter_array = []
    index_array = []
    
    num_params = updated_x.shape[1]
    num_top_points = len(y_rep_dict[method][0])

    top_indices = torch.topk(updated_y.squeeze(), num_top_points).indices

    param_dict = {f'param_{i+1}': [] for i in range(num_params)}

    for i in range(10):
        for rep in range(10):
            value = y_rep_dict[method][i][rep]
            # we save the info for each repeated evaluation of each param set
            method_array.append(method)
            y_value_array.append(value.item())
            top_k_array.append(i+1)
            num_iter_array.append(rep+1)
            index_array.append(top_indices[i].item())
            params = x_rep_dict[method][i]
            for k, v in enumerate(params):
                param_dict[f'param_{k+1}'].append(v.item())

    iter_eval_df = pd.DataFrame({
        'method': method_array,
        'y_value': y_value_array,
        'top_k': top_k_array,
        'num_iter': num_iter_array,
        'index_in_result': index_array,
    })
    for k in range(num_params):
        iter_eval_df[f'param_{k+1}'] = param_dict[f'param_{k+1}']

    return iter_eval_df



def build_repeated_eval_df(x_rep_dict, y_rep_dict):
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

    return repeated_eval_df


def build_avg_eval_df(y_rep_dict):

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
    avg_eval_df['Top-K'] = avg_eval_df['Top-K'].astype(int) + 1

    return avg_eval_df, all_averages


def count_high_values(y_dict, max_iter=200, threshold=0.75):
    name = []
    count_of_value_values = []
    total_points = []
    percentage = []
    all_method = []
    all_hyperparam = []
    for key, value in y_dict.items():
        value = value[:max_iter]
        name.append(key)
        count = torch.sum(value > threshold).item()
        count_of_value_values.append(count)
        total_points.append(value.shape[0])
        percentage.append(round(count / value.shape[0] * 100))
        method, hyperparam = get_method(key)
        all_method.append(method)
        all_hyperparam.append(hyperparam)

    df = pd.DataFrame({'name': name,
                       'count': count_of_value_values,
                       'percentage': percentage,
                       'total_points': total_points,
                       'method': all_method,
                       'hyperparam': all_hyperparam})

    df = df.sort_values(by='count', ascending=False)
    df = df.reset_index(drop=True)
    return df


def get_method(name):
    if 'pop' in name:
        method = 'cma'
        hyperparam = name[:4]
    elif 'depth' in name:
        method = 'ibnn'
        hyperparam = name[:6]
    elif 'bo_' in name:
        method = 'bo'
        hyperparam = 'bo'
    elif 'TR' in name:
        method = 'turbo'
        hyperparam = name[:3]
    else:
        method = 'unknown'
    return method, hyperparam


def get_count_of_high_values_over_threshold(y_dict, unnorm_factor=7000, one_instance_per_method=True):
    comb_df = pd.DataFrame()
    for thresh in range(0, 7000, 100):
        count_df = count_high_values(y_dict, threshold=thresh/unnorm_factor)
        if one_instance_per_method:
            count_df = count_df.groupby('method').max().sort_values(
                by='count', ascending=False).head(10)
        count_df['threshold'] = thresh
        comb_df = pd.concat([comb_df, count_df], axis=0)
    comb_df = comb_df.reset_index(drop=False)
    comb_df = comb_df.sort_values(by=['method', 'threshold'], ascending=True)
    return comb_df
