import os
import warnings
import re

import matplotlib.pyplot as plt
import torch
from torch import nn
import time

from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from botorch import manual_seed
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels import InfiniteWidthBNNKernel
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement


warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}

SMOKE_TEST = os.environ.get("SMOKE_TEST")


def generate_initial_data(f, bounds, n):
    train_x = draw_sobol_samples(bounds=bounds, n=n, q=1).to(**tkwargs)
    train_x = train_x.squeeze(-2) # remove batch dimension

    train_y = []
    for params in train_x:
        current_y = f(params)
        train_y.append(current_y)
    # convert train_y to a [n, 1] tensor
    train_y = torch.tensor(train_y).unsqueeze(-1).to(**tkwargs)

    return train_x, train_y



class MLP(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dims, 50, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(50, 50, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(50, 1, dtype=torch.float64)
        )

    def forward(self, x):
        return self.layers(x)

def create_f(input_dims, seed):
    # create MLP with weights and biases sampled from N(0, 1)
    with manual_seed(seed):
        model = MLP(input_dims).to(**tkwargs)
        params = torch.nn.utils.parameters_to_vector(model.parameters())
        params = torch.randn_like(params, dtype=torch.float64)
        torch.nn.utils.vector_to_parameters(params, model.parameters())

    def f(x):
        with torch.no_grad():
            return model(x)

    return f


def _get_most_recent_result_subfolder_number(results_dir='stored_results', prefix = 'run_'):
    # Get all folders in base_dir that match the pattern run_X
    if not os.path.isdir(results_dir):
        raise FileNotFoundError("The directory 'stored_results' does not exist. Please create it before running the script.")
    else:
        existing = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

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


def get_most_recent_result_subfolder_name(results_dir='stored_results', prefix = 'run_'):
    most_recent_run = _get_most_recent_result_subfolder_number(results_dir, prefix=prefix)
    most_recent_folder = os.path.join(results_dir, f'{prefix}{most_recent_run}')
    return most_recent_folder

def get_new_result_subfolder(results_dir='stored_results', prefix = 'run_'):
    most_recent_run = _get_most_recent_result_subfolder_number(results_dir, prefix=prefix)

    # Determine next folder number
    next_run = most_recent_run + 1
    new_folder = os.path.join(results_dir, f'{prefix}{next_run}')
    os.makedirs(new_folder)

    print(f'Created new subfolder for storing results: {new_folder}')
    return new_folder


def gp_bo_loop(f, bounds, init_x, init_y, kernel, n_iterations, acqf_class, optimize_hypers=False, result_overall_folder='stored_results', seed=0):
    train_x = init_x.clone()
    train_y = init_y.clone()
    start_time = time.time()

    result_folder = os.path.join(result_overall_folder, 'temp_results')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for iteration in range(n_iterations):

        # fit model to data
        model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1), covar_module=kernel)
        if optimize_hypers:
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        model.eval()

        # optimize acquisition function
        candidate_x, acq_value = optimize_acqf(
            acq_function=acqf_class(model, train_y.max()),
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=200,
        )

        # update training points
        train_x = torch.cat([train_x, candidate_x])
        current_y = f(candidate_x)
        current_y = current_y.reshape(1, 1)
        train_y = torch.cat([train_y, current_y])

        # Print iteration runtime
        elapsed = time.time() - start_time
        print(f'Iteration {iteration}: total runtime is {elapsed:.2f} s')
        print('Maximum value so far:', train_y.max().item())

        if iteration == 10:
            # change the result folder to the new one
            result_folder = get_new_result_subfolder()

        if iteration % 1 == 0:
            print('Saved updated_x and updated_y as updated_x.pt and updated_y.pt')
            print('Number of training points:', train_x.size(0))
            torch.save(train_x, os.path.join(result_folder, "updated_x.pt"))
            torch.save(train_y, os.path.join(result_folder, "updated_y.pt"))

        if (iteration % 10 == 0) & (iteration >= 10):
            # save a backup
            print('Saved updated_x and updated_y as updated_x_backup.pt and updated_y_backup.pt')
            torch.save(train_x, os.path.join(result_folder, "updated_x_backup.pt"))
            torch.save(train_y, os.path.join(result_folder, "updated_y_backup.pt"))

        print('================================================================================================')
        print('================================================================================================')

    return train_x, train_y


