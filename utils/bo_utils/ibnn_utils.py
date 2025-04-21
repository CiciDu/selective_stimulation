from utils.bo_utils import shared_utils
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
    train_x = train_x.squeeze(-2)  # remove batch dimension

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


def gp_bo_loop(f, bounds, init_x, init_y, kernel, n_iterations, acqf_class, optimize_hypers=False, result_dir='all_stored_results/ibnn_results', seed=0):

    if optimize_hypers:
        result_dir = os.path.join(result_dir, 'optimize_hypers')

    result_folder = os.path.join(result_dir, 'temp_results')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    train_x = init_x.clone()
    train_y = init_y.clone()
    start_time = time.time()

    for iteration in range(n_iterations):

        # fit model to data
        model = SingleTaskGP(
            train_x, train_y, outcome_transform=Standardize(m=1), covar_module=kernel)
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

        result_folder = shared_utils.save_results_at_fixed_iterations(
            iteration, train_x, train_y, result_dir, result_folder)

        print('================================================================================================')
        print('================================================================================================')

    return train_x, train_y
