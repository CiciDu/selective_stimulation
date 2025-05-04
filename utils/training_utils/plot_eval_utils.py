# code is adapted from https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_Wang_2001.html

from utils.methods_utils import ibnn_utils, shared_utils, turbo_utils, baseline_bo_utils
from utils.sim_utils import set_params_utils, eqs_utils, plotting_utils, obj_func_utils, set_param_space
from utils.methods_utils import ibnn_utils, shared_utils, turbo_utils, baseline_bo_utils, cma_utils

from brian2 import *
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.test_functions import Ackley
from botorch.optim.optimize import optimize_acqf as optimize_acqf_fn
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.models.kernels import InfiniteWidthBNNKernel
from botorch.models.gp_regression import SingleTaskGP as STGP
from botorch.models import SingleTaskGP
from botorch.generation import MaxPosteriorSampling
from botorch.fit import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.acquisition import LogExpectedImprovement, qExpectedImprovement, qLogExpectedImprovement
from botorch import manual_seed
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.constraints import Interval
import gpytorch
from torch.quasirandom import SobolEngine
from torch import nn
import torch
from skopt.plots import plot_gaussian_process, plot_convergence
from skopt.space import Real
from skopt import gp_minimize
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from dataclasses import dataclass
import warnings
import time
import math
import os
import sys
from matplotlib.ticker import MaxNLocator

def plot_both_value_and_cum_max_over_iter(x_dict,
                                          y_dict,
                                          all_methods=None,
                                          unnorm_factor=7000,
                                          max_iter_to_plot=200,
                                          cma_pop_size=8,
                                          show_plot=True,
                                          ):

    # fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_value_over_iter(x_dict, y_dict, all_methods=all_methods,
                         unnorm_factor=unnorm_factor,
                         max_iter_to_plot=max_iter_to_plot,
                         cma_pop_size=cma_pop_size,
                         ax=ax)

    plot_cum_max_over_iter(y_dict, all_methods=all_methods,
                           unnorm_factor=unnorm_factor,
                           max_iter_to_plot=max_iter_to_plot,
                           ax=ax)

    plt.ylabel("Persistent Activity (ms)", fontsize=15)
    plt.title("Objective Values Over Iterations", fontsize=16)
    if show_plot:
        plt.show()
    return


def plot_value_over_iter(x_dict,
                         y_dict,
                         all_methods=None,
                         unnorm_factor=7000,
                         max_iter_to_plot=200,
                         cma_pop_size=8,
                         ax=None,
                         ):

    if all_methods is None:
        all_methods = list(y_dict.keys())

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        show_plot_flag = True
    else:
        show_plot_flag = False
    for method in all_methods:
        y_vals = y_dict[method] * unnorm_factor
        if 'abcd' not in method:
            y_vals = y_vals[:max_iter_to_plot]
            x_axis_values = range(len(y_vals))
        else:
            x_vals = x_dict[method]
            cma_top_y_over_iter, cma_top_params_over_iter, cmu_top_index_over_iter = cma_utils.separate_cma_results_by_pop(
                x_vals, y_vals, cma_pop_size)
            within_bounds = np.where(
                cmu_top_index_over_iter < max_iter_to_plot)[0]
            x_axis_values = cmu_top_index_over_iter[within_bounds]
            y_vals = cma_top_y_over_iter[within_bounds]

        ax.scatter(x_axis_values, y_vals, label=method,
                   alpha=0.6, marker='o', s=25)

    ax.set_xlabel("Iterations", fontsize=15)
    ax.set_ylabel("Persistent Activity (ms)", fontsize=15)
    ax.set_title("Objective Value Over Iterations", fontsize=16)
    # ax.legend(fontsize=15)
    plt.tight_layout()

    if show_plot_flag:
        plt.show()


def plot_cum_max_over_iter(y_dict,
                           all_methods=None,
                           unnorm_factor=7000,
                           max_iter_to_plot=200,
                           ax=None,
                           ):

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        show_plot_flag = True
    else:
        show_plot_flag = False

    if all_methods is None:
        all_methods = list(y_dict.keys())

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    for method in all_methods:
        y_vals = y_dict[method][:max_iter_to_plot] * unnorm_factor
        cum_max = (torch.cummax(y_vals, dim=0)[0]).cpu()
        ax.plot(range(len(cum_max)), cum_max,
                alpha=0.8, linewidth=2.5)

    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("Max Persistent Activity (ms)", fontsize=15)
    # decrease y-axis tick font size
    ax.tick_params(axis='y', labelsize=12)
    ax.set_title("Cumulative Max Objective Value Over Iterations", fontsize=16)
    ax.legend(fontsize=12, bbox_to_anchor=(1.01, 1),
              loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    if show_plot_flag:
        plt.show()


def plot_avg_eval_df(avg_eval_df,
                     all_methods=['baseline', 'turbo', 'ibnn', 'cma'],
                     fr_window=[1100, 8100],
                     top_n_to_plot=10,
                     num_repeats=10):
    """
    Plot the average evaluation DataFrame.

    Args:
        avg_eval_df (pd.DataFrame): DataFrame containing average evaluations.
        fr_window (list): Firing rate window used to compute unnormalized outputs.
        top_n_to_plot (int): Number of top parameter sets to plot.
        num_repeats (int): Number of repeats for the evaluation.
    """

    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot cumulative max or average values across top-k params
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
    # ax.set_ylim(0, fr_window[1] - fr_window[0])
    ax.set_title(
        f"Objective Value for Top Parameters (Average of {num_repeats} Repeats)", fontsize=20, pad=20)

    # Grid and legend
    # ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title="Method", fontsize=15, title_fontsize=16)

    # Optional tweaks
    ax.tick_params(axis='both', labelsize=15)
    fig.tight_layout()
    plt.show()


def plot_top_param_distributions(space, stim_bounds, x_dict, y_dict, num_top_points=5, indices_to_plot=None,
                                 ax=None, show_plot=True, method_to_y_pos_mapping=None, y_tick_label_mapping=None,
                                 param_name_list=None):
    """
    Plot the distribution of top-performing parameter values for each method.

    Args:
        space (list): List of skopt.space.Real objects, each with a `.name` attribute.
        stim_bounds (torch.Tensor): Tensor of shape (2, D) with min and max bounds for each parameter.
        x_dict (dict): Maps method names to torch.Tensor of evaluated parameter values (shape: N x D).
        y_dict (dict): Maps method names to torch.Tensor of objective values (shape: N).
        num_top_points (int): Number of top-performing trials to show per method.
    """

    if param_name_list is None:
        param_name_list = [dim.name for dim in space]

    all_methods = list(x_dict.keys())
    if method_to_y_pos_mapping is None:
        method_to_y_pos_mapping = {method: i for i,
                                   method in enumerate(all_methods)}

    if indices_to_plot is None:
        indices_to_plot = range(len(space))

    for index in indices_to_plot:
        param_name = param_name_list[index]
        bounds = stim_bounds[:, index]
        # print(f"{param_name}: {bounds}")

        if ax is None:
            fig, ax = plt.subplots()

        for method, x_vals in x_dict.items():
            y_vals = y_dict[method]
            top_indices = torch.topk(y_vals.squeeze(), num_top_points).indices
            top_params = x_vals[top_indices]
            top_param_values = top_params[:, index]
            top_param_values = stim_bounds[0, index] + top_param_values * (
                stim_bounds[1, index] - stim_bounds[0, index])

            y_level = method_to_y_pos_mapping[method]
            ax.plot(top_param_values, [
                    y_level] * num_top_points, 'o', markersize=7, 
                    markerfacecolor='none', markeredgewidth=2, alpha=0.85,)

        ax.set_xlim(bounds.tolist())
        #ax.set_title(f"{param_name}: {bounds.tolist()}")
        ax.set_title(f"{param_name}")
        ax.set_yticks(list(method_to_y_pos_mapping.values()))
        if y_tick_label_mapping is not None:
            ax.set_yticklabels(
                [y_tick_label_mapping[value] for value in method_to_y_pos_mapping.values()])
        else:
            ax.set_yticklabels(list(method_to_y_pos_mapping.keys()))

        # delete x and y label
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

        ax.tick_params(axis='x', labelsize=13)
        #ax.tick_params(axis='y', labelsize=13)

        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)

        if show_plot:
            plt.show()
        else:
            return ax


def plot_best_results(updated_x, updated_y, objective_func_tensor, num_top_points=5,
                      fr_window=[1100, 8100]):
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


def plot_count_over_value_thresholds(df, groupby_column='method', columns=['count'], color_map=None,  color_column=None, label_map=None, label_column=None, prefix='',
                                     show_plot=True):
    # Note: column can be ['count', 'percentage']; percentage can be useful because some methods can converge before max_iter is reached

    # df.sort_values(by=[groupby_column, 'threshold'],
    #                ascending=True, inplace=True)
    fig, axes = plt.subplots(1, len(columns), figsize=(len(columns)*7, 5))
    for column in columns:
        if len(columns) > 1:
            ax = axes[columns.index(column)]
        else:
            ax = axes
        for label, group in df.groupby(groupby_column, sort=False):
            if label_map is not None:
                label = label_map[group[label_column].iloc[0]]
            if color_map is not None:
                ax.plot(group['threshold'], group[column], label=label,
                        color=color_map[group[color_column].iloc[0]], alpha=0.8)
            else:
                ax.plot(group['threshold'], group[column],
                        label=label, alpha=0.8)
        ax.set_xlabel('Objective Value Threshold (ms)', fontsize=14)
        ax.set_ylabel(column.capitalize(), fontsize=15)
        ax.set_title(
            f'{prefix}{column.capitalize()} of Values Above Thresholds', fontsize=16)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        # ax.legend(fontsize=12, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        ax.legend(fontsize=14)

    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        return ax
