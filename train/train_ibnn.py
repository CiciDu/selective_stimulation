from utils.training_utils import prep_for_training
from utils.methods_utils import ibnn_utils
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


result_dir = 'all_stored_results/ibnn_results'
if __name__ == "__main__":
    # change to the parent directory
    sys.path.insert(0, os.getcwd())
    # --- Set up initial simulation ---
    from utils.training_utils.set_train_env import *
    result_dir = 'cluster_stored_results/ibnn_results'


# --- define some hyperparameters ---
optimize_hypers = True
N_ITERATIONS = 200
network_depth = 2

# --- get initial data ---
init_x, init_y = prep_for_training.get_init_x_and_y()

# --- Environment settings ---
nrn_options = "-nogui -NSTACK 100000 -NFRAME 20000"
os.environ["NEURON_MODULE_OPTIONS"] = nrn_options
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(237)

# --- define kernels ---
ibnn_kernel = InfiniteWidthBNNKernel(network_depth, device=device)
ibnn_kernel.weight_var = 10.0
ibnn_kernel.bias_var = 5.0
ibnn_kernel = ScaleKernel(ibnn_kernel, device=device)

# --- run BO loop ---
acqf_classes = {"LogEI": LogExpectedImprovement}
results = {}
for acq_name, acqf_class in acqf_classes.items():
    run_bo_with_acqf = partial(ibnn_utils.gp_bo_loop, f=objective_func_tensor, bounds=stim_bounds_norm, acqf_class=acqf_class,
                               result_dir=result_dir,
                               init_x=init_x,
                               init_y=init_y,
                               n_iterations=N_ITERATIONS,
                               )
    ibnn_x, ibnn_y = run_bo_with_acqf(
        kernel=ibnn_kernel, optimize_hypers=optimize_hypers)
    # matern_x, matern_y = run_bo_with_acqf(kernel=matern_kernel, optimize_hypers=True)
    # rbf_x, rbf_y = run_bo_with_acqf(kernel=rbf_kernel, optimize_hypers=True)
    results[acq_name] = {
        "BNN": (ibnn_x, ibnn_y),
        # "Matern": (matern_x, matern_y),
        # "RBF": (rbf_x, rbf_y),
    }
