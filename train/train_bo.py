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


result_dir = 'all_stored_results/bo_results'
if __name__ == "__main__":
    # change to the parent directory
    sys.path.insert(0, os.getcwd())
    # --- Set up initial simulation ---
    from utils.training_utils.set_train_env import *
    result_dir='cluster_stored_results/bo_results'


from utils.training_utils import prep_for_training


init_x, init_y = prep_for_training.get_init_x_and_y()
init_y = - init_y # because we are minimizing
N_ITERATIONS = 200

res = gp_minimize(objective_with_factor, # the function to minimize
                  space_norm,
                x0=init_x.tolist(),
                y0=init_y.reshape(-1).numpy().tolist(),
                  acq_func="EI",      # the acquisition function
                  n_calls=N_ITERATIONS,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise='gaussian',       # the noise level (optional)
                #   random_state=1234 # the random seed
                  )   


train_x, train_y = baseline_bo_utils.save_bo_results(res, result_dir=result_dir)