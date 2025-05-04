import os
import math
import warnings
import re
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.methods_utils import ibnn_utils

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


def retrieve_or_sample_init_points(init_points_folder, objective_func_tensor, stim_bounds_norm, num_init_points=20):
    os.makedirs(init_points_folder, exist_ok=True)
    try:
        updated_x = torch.load(os.path.join(
            init_points_folder, "updated_x.pt"))
        updated_y = torch.load(os.path.join(
            init_points_folder, "updated_y.pt"))
        if updated_x.shape[0] != num_init_points:
            raise ValueError(
                f"Expected {num_init_points} initial points, but got {updated_x.shape[0]}. Will sample new initial points")
    except Exception as e:
        print(f"Error loading initial points: {e}")
        updated_x, updated_y = ibnn_utils.generate_initial_data(
            objective_func_tensor, stim_bounds_norm, n=num_init_points)
        torch.save(updated_x, os.path.join(init_points_folder, "updated_x.pt"))
        torch.save(updated_y, os.path.join(init_points_folder, "updated_y.pt"))
    return updated_x, updated_y


def _get_latest_result_subfolder_number(result_dir='all_stored_results/all_ibnn', prefix='run_'):
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


def get_latest_result_subfolder_name(result_dir='all_stored_results/all_ibnn', prefix='run_'):
    latest_run = _get_latest_result_subfolder_number(
        result_dir, prefix=prefix)
    latest_folder = os.path.join(
        result_dir, f'{prefix}{latest_run}')
    return latest_folder


def get_new_result_subfolder(result_dir='all_stored_results/all_ibnn', prefix='run_'):
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


def save_results(train_x, train_y, result_dir='all_stored_results/all_ibnn'):
    result_folder = get_new_result_subfolder(result_dir)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    torch.save(train_x, os.path.join(result_folder, "updated_x.pt"))
    torch.save(train_y, os.path.join(result_folder, "updated_y.pt"))
    return


def get_latest_results(result_dir, prefix='run_'):
    # try directly loading the files
    # If it fails, get the latest folder and load from there

    try:
        updated_x = torch.load(os.path.join(result_dir, "updated_x.pt"))
        updated_y = torch.load(os.path.join(result_dir, "updated_y.pt"))
        return updated_x, updated_y
    except Exception as e:
        latest_folder = get_latest_result_subfolder_name(
            result_dir=result_dir, prefix=prefix)
        updated_x = torch.load(os.path.join(latest_folder, "updated_x.pt"))
        updated_y = torch.load(os.path.join(latest_folder, "updated_y.pt"))
    return updated_x, updated_y


def load_top_optimization_results(
    results_dir_dict,
    prefix='run_',
    verbose=True,
):
    x_dict, y_dict = {}, {}

    for method, result_dir in results_dir_dict.items():
        try:
            updated_x, updated_y = get_latest_results(
                result_dir, prefix=prefix)

            x_dict[method] = updated_x
            y_dict[method] = updated_y
            if verbose:
                print(f"Loaded {method}")
        except Exception as e:
            if verbose:
                print(f"Failed to load results for method '{method}': {e}")

    return x_dict, y_dict


def check_all_results_exist(results_dir_dict, verbose=True):
    print('================================================================================================')
    all_methods = results_dir_dict.keys()
    for method in all_methods:
        result_dir = results_dir_dict[method]
        try:
            updated_x, updated_y = get_latest_results(result_dir)
            if verbose:
                print(f"Method exists: {method}")
        except Exception as e:
            if verbose:
                print(f"Failed to load results for method '{method}': {e}")
        print('================================================================================================')
    return


def get_all_dir_from_method(category):
    method_dir = f'all_stored_results/all_{category}/'
    folders = [f for f in os.listdir(
        method_dir) if os.path.isdir(os.path.join(method_dir, f))]
    results_dir_dict = {}
    for folder in folders:
        results_dir_dict[folder] = os.path.join(method_dir, folder)
    print(f"Found {results_dir_dict.keys()}")
    return results_dir_dict


def get_all_results_dir_dict(base_dir='all_stored_results'):
    """
    Get all results directory dictionary for different methods.
    """
    # Define the base directory
    

    # Check if the base directory exists
    if not os.path.exists(base_dir):
        raise FileNotFoundError(
            f"The base directory '{base_dir}' does not exist. Please create it before running the script.")

    # Initialize an empty dictionary to store results directories
    all_results_dir_dict = {}
    for subfolder in ['all_cma', 'all_bo', 'all_ibnn', 'all_turboM']:
        method_dir = os.path.join(base_dir, subfolder)
        folders = [f for f in os.listdir(method_dir) if os.path.isdir(os.path.join(method_dir, f))]
        for folder in folders:
            all_results_dir_dict[folder] = os.path.join(method_dir, folder)
    return all_results_dir_dict
            