from utils.training_utils import prep_for_training
import sys
import os
import warnings
import torch
import warnings
import numpy as np
from functools import partial
from gpytorch.kernels import ScaleKernel
from botorch.acquisition import LogExpectedImprovement
from botorch.models.kernels import InfiniteWidthBNNKernel

print("Current directory:", os.getcwd())


result_dir = 'all_stored_results/turbo_results'
if __name__ == "__main__":
    # change to the parent directory
    sys.path.insert(0, os.getcwd())
    # --- Set up initial simulation ---
    from utils.training_utils.set_train_env import *
    result_dir = 'cluster_stored_results/turbo_results'


batch_size = 2
N_ITERATIONS = 200
max_cholesky_size = float("inf")  # Always use Cholesky

init_x, init_y = prep_for_training.get_init_x_and_y()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

state = turbo_utils.TurboState(dim=dim, batch_size=batch_size)

X_turbo, Y_turbo = turbo_utils.run_turbo_loop(init_x, init_y, objective_func_tensor, dim,
                                              max_iter=N_ITERATIONS, batch_size=batch_size,
                                              max_cholesky_size=max_cholesky_size, device=device, dtype=dtype,
                                              SMOKE_TEST=False,
                                              result_dir='all_stored_results/turbo_results'
                                              )
